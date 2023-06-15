import itertools
from typing import Any, Callable, Dict, Optional, Union, List

import spacy
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteCrossAttnProcessor,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor

from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, \
    align_wordpieces_indices, extract_attribution_indices

logger = logging.get_logger(__name__)


class SynGenDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPFeatureExtractor,
                 requires_safety_checker: bool = True,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

        self.parser = spacy.load("en_core_web_trf")

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = _get_attention_maps_list(
            attention_maps=attention_maps
        )
        return attention_maps_list

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    # Based on StableDiffusionPipeline.__call__ . New code is annotated with NEW.
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            syngen_step_size: float = 20.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            syngen_step_size (`float`, *optional*, default to 20.0):
                Controls the step size of each SynGen update.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # NEW - stores the attention calculated in the unet
        self.attention_store = AttentionStore()
        self.register_attention_control()

        # NEW
        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt:] if do_classifier_free_guidance else prompt_embeds
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # NEW
                latents = self._syngen_step(
                    latents,
                    text_embeddings,
                    t,
                    i,
                    syngen_step_size,
                    cross_attention_kwargs,
                    prompt,
                    max_iter_to_alter=25,
                )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def _syngen_step(
            self,
            latents,
            text_embeddings,
            t,
            i,
            step_size,
            cross_attention_kwargs,
            prompt,
            max_iter_to_alter=25,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                text_embedding = text_embedding.unsqueeze(0)

                self.unet(
                    latent,
                    t,
                    encoder_hidden_states=text_embedding,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                self.unet.zero_grad()

                # Get attention maps
                attention_maps = self._aggregate_and_get_attention_maps_per_token()

                loss = self._compute_loss(attention_maps=attention_maps, prompt=prompt)

                # Perform gradient update
                if i < max_iter_to_alter:
                    if loss != 0:
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=step_size
                        )
                    logger.info(f"Iteration {i} | Loss: {loss:0.4f}")

            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents

    def _compute_loss(
            self, attention_maps: List[torch.Tensor], prompt: Union[str, List[str]]
    ) -> torch.Tensor:
        attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(self.tokenizer, prompt)
        loss = self._attribution_loss(attention_maps, prompt, attn_map_idx_to_wp)

        return loss


    def _attribution_loss(
            self,
            attention_maps: List[torch.Tensor],
            prompt: Union[str, List[str]],
            attn_map_idx_to_wp,
    ) -> torch.Tensor:
        subtrees_indices = self._extract_attribution_indices(prompt)
        loss = 0

        for subtree_indices in subtrees_indices:
            noun, modifier = split_indices(subtree_indices)
            all_subtree_pairs = list(itertools.product(noun, modifier))
            positive_loss, negative_loss = self._calculate_losses(
                attention_maps,
                all_subtree_pairs,
                subtree_indices,
                attn_map_idx_to_wp,
            )
            loss += positive_loss
            loss += negative_loss

        return loss

    def _calculate_losses(
            self,
            attention_maps,
            all_subtree_pairs,
            subtree_indices,
            attn_map_idx_to_wp,
    ):
        positive_loss = []
        negative_loss = []
        for pair in all_subtree_pairs:
            noun, modifier = pair
            positive_loss.append(
                calculate_positive_loss(attention_maps, modifier, noun)
            )
            negative_loss.append(
                calculate_negative_loss(
                    attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
                )
            )

        positive_loss = sum(positive_loss)
        negative_loss = sum(negative_loss)

        return positive_loss, negative_loss

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text == wp:
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.startswith(wp) and wp != member.text:  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all([wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            paired_indices.append(curr_collected_wp_indices)
            
        return paired_indices

    def _extract_attribution_indices(self, prompt):
        pairs = extract_attribution_indices(prompt, self.parser)
        paired_indices = self._align_indices(prompt, pairs)
        return paired_indices



def _get_attention_maps_list(
        attention_maps: torch.Tensor
) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list
