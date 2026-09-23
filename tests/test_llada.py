import unittest
import copy

import torch

from llada.data import CharTokenizer
from llada.model import LLaDA, LLaDAConfig, resize_token_embeddings
from llada.objective import (
    llada_pretraining_loss,
    llada_sft_loss,
    make_diffusion_batch,
    make_sft_diffusion_batch,
)
from llada.sampling import generate
from llada.reward import CharacterNGramScorer, character_ngram_f1, shakespeare_reward
from llada.sampling import generate_blockwise
from llada.sft_data import parse_dialogue_pairs
from llada.spg import make_blockwise_perturbations, spg_scores


class LLaDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = LLaDAConfig(
            vocab_size=17,
            max_seq_len=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            ffn_dim=32,
        )
        self.model = LLaDA(self.config)

    def test_forward_and_pretraining_step(self):
        clean = torch.randint(0, 16, (4, 12))
        batch = make_diffusion_batch(clean, mask_id=16)
        logits = self.model(batch.noisy_tokens)
        self.assertEqual(logits.shape, (4, 12, 17))
        loss, metrics = llada_pretraining_loss(logits, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(metrics["mask_fraction"].item(), 0.0)
        loss.backward()
        self.assertIsNotNone(self.model.token_embedding.weight.grad)

    def test_attention_is_not_causal(self):
        self.model.eval()
        first = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 3, 5]])
        with torch.no_grad():
            logits_a = self.model(first)
            logits_b = self.model(changed_future)
        self.assertFalse(torch.allclose(logits_a[:, 0], logits_b[:, 0]))

    def test_low_confidence_generation_resolves_masks_and_keeps_prompt(self):
        prompt = torch.tensor([[1, 2, 3]])
        output = generate(
            self.model,
            mask_id=16,
            prompt_tokens=prompt,
            generation_length=8,
            steps=4,
            remasking="low_confidence",
        )
        self.assertEqual(output.shape, (1, 11))
        self.assertTrue(torch.equal(output[:, :3], prompt))
        self.assertFalse(output[:, 3:].eq(16).any())

    def test_random_generation_resolves_masks(self):
        prompt = torch.empty((2, 0), dtype=torch.long)
        output = generate(
            self.model,
            mask_id=16,
            prompt_tokens=prompt,
            generation_length=8,
            steps=4,
            remasking="random",
            temperature=0.8,
            top_k=5,
        )
        self.assertEqual(output.shape, (2, 8))
        self.assertFalse(output.eq(16).any())

    def test_sft_masks_only_response_and_computes_loss(self):
        clean = torch.randint(0, 16, (3, 10))
        response_positions = torch.zeros_like(clean, dtype=torch.bool)
        response_positions[:, 4:] = True
        batch = make_sft_diffusion_batch(clean, response_positions, mask_id=16)
        self.assertTrue(torch.equal(batch.noisy_tokens[:, :4], clean[:, :4]))
        self.assertFalse(batch.masked_positions[:, :4].any())
        logits = self.model(batch.noisy_tokens)
        loss, _ = llada_sft_loss(logits, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_resize_vocabulary_preserves_pretrained_rows(self):
        old_embedding = self.model.token_embedding.weight.detach().clone()
        resized = resize_token_embeddings(self.model, 18)
        self.assertEqual(resized.config.vocab_size, 18)
        self.assertTrue(torch.equal(resized.token_embedding.weight[:17], old_embedding))
        self.assertIs(resized.token_embedding.weight, resized.lm_head.weight)

    def test_dialogue_parser_and_eos_tokenizer(self):
        text = "ROMEO:\nHello.\n\nJULIET:\nFarewell.\n"
        pairs = parse_dialogue_pairs(text, context_turns=1)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[1].prompt, "ROMEO:\nHello.\n\nJULIET:\n")
        tokenizer = CharTokenizer(tuple(sorted(set(text))), has_eos=True)
        self.assertEqual(tokenizer.vocab_size, len(tokenizer.chars) + 2)
        self.assertEqual(tokenizer.decode([tokenizer.eos_id]), "")

    def test_blockwise_generation(self):
        prompt = torch.tensor([[1, 2], [1, 2]])
        output = generate_blockwise(
            self.model,
            mask_id=16,
            prompt_tokens=prompt,
            generation_length=8,
            steps=4,
            block_length=4,
            temperature=0.8,
            top_k=5,
        )
        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(output.eq(16).any())

    def test_spg_bounds_are_finite_and_differentiable(self):
        clean = torch.randint(0, 16, (3, 10))
        completion = torch.ones((3, 8), dtype=torch.bool)
        perturbations = make_blockwise_perturbations(
            clean,
            prompt_length=2,
            completion_mask=completion,
            mask_id=16,
            block_length=4,
            mc_samples=2,
            prompt_mask_probability=0.0,
        )
        self.assertEqual(perturbations.noisy_tokens.shape, (3, 2, 10))
        self.assertTrue(perturbations.active_response_masks.any(dim=-1).all())
        reference = copy.deepcopy(self.model).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        elbo, eubo, mixed, kl = spg_scores(
            self.model,
            reference,
            clean,
            perturbations,
            prompt_length=2,
            completion_mask=completion,
            eubo_beta=1.5,
            mixture_weight=0.5,
        )
        self.assertTrue(torch.isfinite(elbo).all())
        self.assertTrue(torch.isfinite(eubo).all())
        loss = -(elbo.mean() + mixed.mean()) + 0.02 * kl
        loss.backward()

    def test_shakespeare_reward_components(self):
        corpus = "To be, or not to be. That is the question. " * 4
        scorer = CharacterNGramScorer(corpus)
        exact = character_ngram_f1("To be", "To be")
        mismatch = character_ngram_f1("xyz", "To be")
        self.assertGreater(exact, mismatch)
        reward = shakespeare_reward(
            "To be, or not to be.",
            "To be, or not to be.",
            fluency_scorer=scorer,
            has_eos=True,
            generation_length=32,
        )
        self.assertGreaterEqual(reward.total, 0.0)
        self.assertLessEqual(reward.total, 1.0)


if __name__ == "__main__":
    unittest.main()
