"""
DeepInterestGR - Deep Interest Mining for Generative Recommendation

This script implements the core components of DeepInterestGR:
1. Multi-LLM Interest Mining (MLIM)
2. Reward-Labeled Deep Interest (RLDI)
3. Interest-Enhanced Item Discretization (IEID)
4. Interest-Aware Reward for RL optimization

Based on the paper: DeepInterestGR: Mining Deep Multi-Interest Using Multi-Modal LLMs for Generative Recommendation
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel


class MultiLLMInterestMining:
    """
    Multi-LLM Interest Mining (MLIM) module.
    Extracts deep interests from item metadata using multiple LLMs.
    """
    
    def __init__(self, llm_providers: List[str] = None):
        """
        Initialize MLIM with supported LLM providers.
        
        Args:
            llm_providers: List of LLM providers to use (e.g., "gpt", "gemini", "kimi", "grok")
        """
        self.providers = llm_providers or ["gpt", "gemini", "kimi", "grok"]
        self.interest_prompt_template = """
        Given the following product information, extract deep user interests:
        
        Title: {title}
        Description: {description}
        
        Step 1: Identify surface-level attributes
        Step 2: Infer latent motivations and lifestyle preferences
        Step 3: Generate interpretable interest tags with confidence scores
        
        Output format:
        [Interest_1]: {text} | Confidence: {high/medium/low}
        [Interest_2]: {text} | Confidence: {high/medium/low}
        [Lifestyle]: {one_sentence_profile}
        """
    
    def extract_interests(self, item_title: str, item_description: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract deep interests from item metadata using multiple LLMs.
        
        Args:
            item_title: Item title
            item_description: Item description
            
        Returns:
            Dictionary mapping LLM provider to list of (interest, confidence) tuples
        """
        interests = {}
        
        for provider in self.providers:
            prompt = self.interest_prompt_template.format(
                title=item_title,
                description=item_description
            )
            # In real implementation, this would call the actual LLM API
            # For demonstration, we simulate the response
            interests[provider] = self._simulate_llm_response(provider, item_title)
        
        return interests
    
    def _simulate_llm_response(self, provider: str, title: str) -> List[Tuple[str, float]]:
        """Simulate LLM response for demonstration."""
        base_interests = [
            ("Quality-conscious", 0.85),
            ("Trend-following", 0.78),
            ("Value-seeking", 0.72),
            ("Convenience-focused", 0.68),
        ]
        
        # Add provider-specific variations
        provider_bonus = {
            "gpt": [("Tech-savvy", 0.82), ("Premium buyer", 0.75)],
            "gemini": [("Eco-conscious", 0.80), ("Brand loyal", 0.73)],
            "kimi": [("Fashion-forward", 0.79), ("Social influencer", 0.71)],
            "grok": [("Budget-friendly", 0.77), ("Novelty-seeker", 0.70)],
        }
        
        return base_interests + provider_bonus.get(provider, [])
    
    def aggregate_interests(self, interests: Dict[str, List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """
        Aggregate interests from multiple LLMs using consensus-based merging.
        
        Args:
            interests: Raw interests from each LLM
            
        Returns:
            Aggregated list of (interest, confidence) tuples
        """
        interest_scores = {}
        
        for provider, provider_interests in interests.items():
            for interest, confidence in provider_interests:
                if interest not in interest_scores:
                    interest_scores[interest] = []
                interest_scores[interest].append(confidence)
        
        # Compute aggregated confidence (average + frequency bonus)
        aggregated = []
        for interest, scores in interest_scores.items():
            avg_confidence = np.mean(scores)
            frequency_bonus = min(len(scores) / len(self.providers), 0.3)
            aggregated_confidence = min(avg_confidence + frequency_bonus, 1.0)
            aggregated.append((interest, aggregated_confidence))
        
        # Sort by confidence
        aggregated.sort(key=lambda x: -x[1])
        
        return aggregated


class RewardLabeledDeepInterest:
    """
    Reward-Labeled Deep Interest (RLDI) module.
    Assigns quality labels to mined interests for RL supervision.
    """
    
    def __init__(self, classifier_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize RLDI with a quality classifier."""
        self.classifier_model = classifier_model
        self.classification_prompt = """
        Classify the following interest as 1 (positive) or 0 (negative):
        
        Interest: "{interest}"
        Source Items: {items}
        
        Label = 1 if: Specific + Actionable + Authentic
        Label = 0 if: Vague / Generic / Hallucinated / Contradictory
        
        Output: Label: {0/1}
        """
    
    def classify_interest(self, interest: str, source_items: List[str]) -> int:
        """
        Classify an interest as positive (1) or negative (0).
        
        Args:
            interest: The interest description
            source_items: Items that generated this interest
            
        Returns:
            1 for positive, 0 for negative
        """
        # In real implementation, this would call the classifier LLM
        # For demonstration, we use simple heuristic rules
        
        # Check for generic/vague terms
        vague_terms = ["general", "various", "different", "many", "some", "thing"]
        if any(term in interest.lower() for term in vague_terms):
            return 0
        
        # Check for specificity (minimum length and content)
        if len(interest.split()) < 2:
            return 0
        
        # Check for actionable content
        actionable_terms = ["focused", "seeking", "conscious", "forward", "friendly", "savvy"]
        if any(term in interest.lower() for term in actionable_terms):
            return 1
        
        return 1  # Default to positive for demonstration


class InterestEnhancedItemDiscretization:
    """
    Interest-Enhanced Item Discretization (IEID) module.
    Encodes deep interests into semantic embeddings and quantizes to SIDs.
    """
    
    def __init__(self, embedding_model: str = "Qwen/Qwen3-Embedding-4B"):
        """Initialize IEID with an embedding model."""
        self.embedding_model = embedding_model
        self.sid_length = 4  # Number of quantization layers
        self.codebook_size = 256  # Codebook size per layer
        
        # Simulated codebooks for demonstration
        self.codebooks = [
            np.random.randn(self.codebook_size, 768) for _ in range(self.sid_length)
        ]
    
    def encode_interests(self, interests: List[str]) -> np.ndarray:
        """
        Encode interest descriptions into semantic embeddings.
        
        Args:
            interests: List of interest descriptions
            
        Returns:
            Concatenated embedding vector
        """
        # In real implementation, this would use Qwen3-Embedding-4B
        # For demonstration, generate random embeddings
        embeddings = []
        for interest in interests:
            # Simulate embedding based on interest text
            embedding = np.random.randn(768) * 0.1
            embedding[hash(interest) % 768] += 1.0  # Add some content-based variation
            embeddings.append(embedding)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.random.randn(768)
    
    def quantize_to_sid(self, embedding: np.ndarray) -> Tuple[int, ...]:
        """
        Quantize embedding to Semantic ID (SID) using residual quantization.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Tuple of SID tokens (one per quantization layer)
        """
        residual = embedding.copy()
        sid_tokens = []
        
        for layer in range(self.sid_length):
            # Find closest codebook entry
            codebook = self.codebooks[layer]
            distances = np.linalg.norm(codebook - residual, axis=1)
            token = int(np.argmin(distances))
            sid_tokens.append(token)
            
            # Update residual
            residual = residual - codebook[token]
        
        return tuple(sid_tokens)
    
    def process_item(self, item_title: str, interests: List[str]) -> Tuple[int, ...]:
        """
        End-to-end IEID processing for an item.
        
        Args:
            item_title: Item title for reference
            interests: List of deep interests for the item
            
        Returns:
            SID tuple for the item
        """
        embedding = self.encode_interests(interests)
        return self.quantize_to_sid(embedding)


class InterestAwareReward:
    """
    Interest-Aware Reward module.
    Computes rewards based on interest alignment for RL optimization.
    """
    
    def __init__(self, rldi_labels: Dict[str, int]):
        """
        Initialize with RLDI labels.
        
        Args:
            rldi_labels: Dictionary mapping interest to quality label (0 or 1)
        """
        self.rldi_labels = rldi_labels
        self.alpha = 0.7  # Weight for interest alignment
        self.beta = 0.3   # Weight for relevance
    
    def compute_reward(self, generated_sid: Tuple[int, ...], user_interests: List[str], 
                       ground_truth_sid: Optional[Tuple[int, ...]] = None) -> float:
        """
        Compute the Interest-Aware Reward for a generated recommendation.
        
        Args:
            generated_sid: Generated SID sequence
            user_interests: User's deep interests
            ground_truth_sid: Optional ground truth SID for relevance reward
            
        Returns:
            Combined reward score
        """
        # Compute interest alignment reward
        interest_reward = self._compute_interest_alignment(user_interests)
        
        # Compute relevance reward (if ground truth available)
        relevance_reward = 0.0
        if ground_truth_sid is not None:
            relevance_reward = self._compute_relevance(generated_sid, ground_truth_sid)
        
        # Combine rewards
        total_reward = self.alpha * interest_reward + self.beta * relevance_reward
        
        return total_reward
    
    def _compute_interest_alignment(self, interests: List[str]) -> float:
        """Compute reward based on positive interest alignment."""
        if not interests:
            return 0.5  # Neutral reward
        
        positive_count = sum(
            1 for interest in interests 
            if self.rldi_labels.get(interest, 1) == 1
        )
        
        return positive_count / len(interests)
    
    def _compute_relevance(self, generated: Tuple[int, ...], ground_truth: Tuple[int, ...]) -> float:
        """Compute relevance reward based on SID similarity."""
        # Simple matching score
        matches = sum(g == gt for g, gt in zip(generated, ground_truth))
        return matches / len(generated)


class DeepInterestGRTrainer:
    """
    DeepInterestGR two-stage trainer: SFT + RL with Interest-Aware Reward.
    Implements the complete training pipeline for generative recommendation.
    """
    
    def __init__(self, reward_module: InterestAwareReward):
        """
        Initialize trainer with Interest-Aware Reward module.
        
        Args:
            reward_module: InterestAwareReward instance for computing rewards
        """
        self.reward_module = reward_module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training hyperparameters
        self.sft_lr = 3e-4
        self.rl_lr = 1e-5
        self.sft_epochs = 3
        self.rl_epochs = 2
        self.batch_size = 256
        self.temperature = 0.5
        self.num_generations = 10
        self.kl_coeff = 0.001
        self.beta = 0.04  # GRPO beta parameter
    
    def sft_train(self, model, train_data):
        """
        Supervised Fine-Tuning (SFT) stage.
        
        Args:
            model: Generative model to train
            train_data: Training dataset
            
        Returns:
            Trained model after SFT
        """
        print("\n=== Starting SFT Training ===")
        
        # Simulate SFT training
        for epoch in range(self.sft_epochs):
            total_loss = 0.0
            num_batches = len(train_data) // self.batch_size
            
            for batch_idx in range(num_batches):
                # Simulate forward pass and loss computation
                loss = self._simulate_sft_step()
                total_loss += loss
                
                if batch_idx % 10 == 0:
                    print(f"  SFT Epoch {epoch+1}/{self.sft_epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"  SFT Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")
        
        print("=== SFT Training Complete ===\n")
        return model
    
    def rl_train(self, model, ref_model, train_data, user_interests):
        """
        Reinforcement Learning (RL) stage with GRPO.
        
        Args:
            model: Policy model to optimize
            ref_model: Reference model (from SFT)
            train_data: Training dataset
            user_interests: User deep interests for reward computation
            
        Returns:
            Optimized policy model
        """
        print("\n=== Starting RL Training with GRPO ===")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.rl_lr)
        
        for epoch in range(self.rl_epochs):
            total_reward = 0.0
            total_kl = 0.0
            num_batches = len(train_data) // self.batch_size
            
            for batch_idx in range(num_batches):
                # Generate multiple responses per prompt (GRPO group)
                rewards = []
                kl_divs = []
                
                for _ in range(self.num_generations):
                    # Generate SID sequence
                    generated_sid = self._generate_sid(model)
                    ground_truth_sid = self._get_ground_truth_sid(train_data)
                    
                    # Compute reward
                    reward = self.reward_module.compute_reward(
                        generated_sid,
                        user_interests,
                        ground_truth_sid
                    )
                    rewards.append(reward)
                    
                    # Compute KL divergence from reference model
                    kl_div = self._compute_kl_divergence(model, ref_model)
                    kl_divs.append(kl_div)
                
                # GRPO: Group-normalized advantage computation
                normalized_rewards = self._normalize_rewards(rewards)
                
                # Compute policy gradient loss
                loss = self._compute_grpo_loss(
                    normalized_rewards,
                    kl_divs,
                    self.kl_coeff
                )
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_reward += sum(rewards) / self.num_generations
                total_kl += sum(kl_divs) / self.num_generations
                
                if batch_idx % 10 == 0:
                    print(f"  RL Epoch {epoch+1}/{self.rl_epochs}, Batch {batch_idx}/{num_batches}")
                    print(f"    Avg Reward: {sum(rewards)/self.num_generations:.4f}")
                    print(f"    Avg KL: {sum(kl_divs)/self.num_generations:.4f}")
            
            avg_reward = total_reward / num_batches
            avg_kl = total_kl / num_batches
            print(f"  RL Epoch {epoch+1} Complete - Avg Reward: {avg_reward:.4f}, Avg KL: {avg_kl:.4f}")
        
        print("=== RL Training Complete ===\n")
        return model
    
    def _simulate_sft_step(self):
        """Simulate a single SFT training step."""
        return np.random.uniform(0.1, 0.5)  # Simulated loss
    
    def _generate_sid(self, model):
        """Simulate SID generation."""
        return tuple(np.random.randint(0, 256, 4))
    
    def _get_ground_truth_sid(self, train_data):
        """Simulate getting ground truth SID from data."""
        return tuple(np.random.randint(0, 256, 4))
    
    def _compute_kl_divergence(self, model, ref_model):
        """Simulate KL divergence computation."""
        return np.random.uniform(0.001, 0.01)
    
    def _normalize_rewards(self, rewards):
        """Normalize rewards within a group (GRPO)."""
        rewards = np.array(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8
        return (rewards - mean) / std
    
    def _compute_grpo_loss(self, normalized_rewards, kl_divs, kl_coeff):
        """Compute GRPO loss with KL regularization."""
        # GRPO loss: -E[normalized_reward * log_prob] + beta * KL
        policy_loss = -np.mean(normalized_rewards) * 0.1  # Simulated
        kl_loss = kl_coeff * np.mean(kl_divs)
        return torch.tensor(policy_loss + kl_loss, requires_grad=True)


def main():
    """
    Example workflow for DeepInterestGR.
    Demonstrates the complete pipeline from interest mining to two-stage training.
    """
    print("=== DeepInterestGR Pipeline Demo ===")
    
    # Example item
    item = {
        "title": "Premium Wireless Noise-Canceling Headphones",
        "description": "High-fidelity audio with advanced active noise cancellation. "
                      "Perfect for frequent travelers and remote workers seeking focus.",
    }
    
    # Step 1: Multi-LLM Interest Mining
    print("\n1. Multi-LLM Interest Mining (MLIM)")
    mlim = MultiLLMInterestMining()
    raw_interests = mlim.extract_interests(item["title"], item["description"])
    
    print("Raw interests from each LLM:")
    for provider, interests in raw_interests.items():
        print(f"  {provider}: {interests[:2]}...")
    
    aggregated_interests = mlim.aggregate_interests(raw_interests)
    print(f"\nAggregated interests: {aggregated_interests}")
    
    # Step 2: Reward-Labeled Deep Interest (RLDI)
    print("\n2. Reward-Labeled Deep Interest (RLDI)")
    rldi = RewardLabeledDeepInterest()
    rldi_labels = {}
    
    for interest, confidence in aggregated_interests:
        label = rldi.classify_interest(interest, [item["title"]])
        rldi_labels[interest] = label
        print(f"  {interest}: Label={label} (confidence={confidence:.2f})")
    
    # Step 3: Interest-Enhanced Item Discretization (IEID)
    print("\n3. Interest-Enhanced Item Discretization (IEID)")
    ieid = InterestEnhancedItemDiscretization()
    interest_texts = [interest for interest, _ in aggregated_interests]
    sid = ieid.process_item(item["title"], interest_texts)
    print(f"  Generated SID: {sid}")
    
    # Step 4: Interest-Aware Reward
    print("\n4. Interest-Aware Reward Computation")
    reward_module = InterestAwareReward(rldi_labels)
    
    # Simulate a generated recommendation
    generated_sid = (128, 64, 192, 32)  # Simulated generated SID
    ground_truth_sid = sid  # Use actual item SID as ground truth
    
    reward = reward_module.compute_reward(
        generated_sid,
        interest_texts,
        ground_truth_sid
    )
    print(f"  Reward for generated SID {generated_sid}: {reward:.4f}")
    
    # Step 5: Two-Stage Training (SFT + RL)
    print("\n5. Two-Stage Training (SFT + RL)")
    trainer = DeepInterestGRTrainer(reward_module)
    
    # Simulate models
    model = "Qwen2.5-7B-Instruct (Simulated)"
    ref_model = "Qwen2.5-7B-Instruct (Reference)"
    train_data = list(range(1000))  # Simulated dataset
    
    # SFT Training
    model = trainer.sft_train(model, train_data)
    
    # RL Training with GRPO
    model = trainer.rl_train(model, ref_model, train_data, interest_texts)
    
    print("\n=== DeepInterestGR Pipeline Complete ===")


if __name__ == "__main__":
    main()
