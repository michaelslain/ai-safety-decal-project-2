"""
adversarial example generation for nlp
demonstrates adversarial attacks on sentiment analysis models
works on m1 mac with transformers 4.42.4 (avoids bus error in newer versions)
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import Dict, List
import re


class AdversarialTextAttack:
    """class for generating and analyzing adversarial text examples"""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        initialize the adversarial attack system
        model_name: huggingface model to attack
        """
        self.model_name = model_name
        self.device = torch.device("cpu")  # use cpu to avoid mps issues on m1
        print(f"Using device: {self.device}")

        # load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.attack_results = []

    def predict(self, text: str) -> Dict:
        """
        make a prediction on a text sample
        text: input text to classify
        returns: dictionary with prediction and confidence
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                               padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        # get label
        label_map = self.model.config.id2label
        label = label_map[pred_class]

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probs[0].cpu().tolist()
        }

    def word_substitution_attack(self, text: str, substitutions: Dict[str, List[str]]) -> List[Dict]:
        """
        perform word substitution attack by trying different synonym replacements
        text: original text
        substitutions: dictionary mapping words to potential substitutes
        returns: list of successful adversarial examples
        """
        orig_pred = self.predict(text)
        orig_label = orig_pred["label"]
        successful_attacks = []

        # try each substitution
        for word, candidates in substitutions.items():
            for candidate in candidates:
                if word.lower() in text.lower():
                    # create adversarial text
                    adv_text = re.sub(r'\b' + re.escape(word) + r'\b', candidate, text, flags=re.IGNORECASE)
                    adv_pred = self.predict(adv_text)

                    # check if attack succeeded
                    if adv_pred["label"] != orig_label:
                        successful_attacks.append({
                            "original_text": text,
                            "adversarial_text": adv_text,
                            "word_changed": word,
                            "substitute": candidate,
                            "original_prediction": orig_pred,
                            "adversarial_prediction": adv_pred,
                            "success": True
                        })
                        return successful_attacks  # return first successful attack

        return successful_attacks

    def character_level_attack(self, text: str, target_words: List[str]) -> List[Dict]:
        """
        perform character-level perturbation attack
        text: original text
        target_words: words to perturb
        returns: list of successful adversarial examples
        """
        orig_pred = self.predict(text)
        orig_label = orig_pred["label"]
        successful_attacks = []

        for word in target_words:
            if word.lower() in text.lower():
                # try different character-level perturbations
                perturbations = [
                    (word, word[:-1]),  # delete last character
                    (word, word[0] + word[2:] if len(word) > 2 else word),  # delete second character
                    (word, word[0] + word[1]*2 + word[2:] if len(word) > 2 else word),  # duplicate character
                ]

                for original, perturbed in perturbations:
                    adv_text = re.sub(r'\b' + re.escape(original) + r'\b', perturbed, text, flags=re.IGNORECASE)
                    adv_pred = self.predict(adv_text)

                    if adv_pred["label"] != orig_label:
                        successful_attacks.append({
                            "original_text": text,
                            "adversarial_text": adv_text,
                            "word_changed": original,
                            "perturbation": perturbed,
                            "original_prediction": orig_pred,
                            "adversarial_prediction": adv_pred,
                            "success": True
                        })
                        return successful_attacks

        return successful_attacks

    def run_comprehensive_attack(self, text: str, attack_type: str = "word_substitution") -> Dict:
        """
        run comprehensive attack trying multiple strategies
        """
        print(f"\nAttacking: \"{text}\"")
        orig_pred = self.predict(text)
        print(f"Original prediction: {orig_pred['label']} (confidence: {orig_pred['confidence']:.3f})")

        if attack_type == "word_substitution":
            # define strategic substitutions that are likely to flip sentiment
            substitutions = {
                "wonderful": ["terrible", "awful", "horrible", "dreadful"],
                "loved": ["hated", "despised", "loathed"],
                "delicious": ["disgusting", "awful", "terrible"],
                "happy": ["disappointed", "unhappy", "upset"],
                "exceeded": ["failed", "missed"],
                "terrible": ["wonderful", "great", "excellent"],
                "waste": ["use", "pleasure"],
                "awful": ["great", "wonderful", "excellent"],
                "hate": ["love", "enjoy"],
            }

            results = self.word_substitution_attack(text, substitutions)

        else:  # character_level
            target_words = ["wonderful", "loved", "delicious", "happy", "terrible", "waste", "awful", "hate"]
            results = self.character_level_attack(text, target_words)

        if results:
            result = results[0]
            print(f"✓ Attack SUCCESSFUL!")
            print(f"Adversarial text: \"{result['adversarial_text']}\"")
            print(f"New prediction: {result['adversarial_prediction']['label']} "
                  f"(confidence: {result['adversarial_prediction']['confidence']:.3f})")

            # add to results
            self.attack_results.append({
                "attack_type": attack_type,
                "success": True,
                "original_text": text,
                "adversarial_text": result["adversarial_text"],
                "original_prediction": orig_pred,
                "adversarial_prediction": result["adversarial_prediction"],
                "modification": result.get("word_changed", result.get("perturbation", ""))
            })
            return result
        else:
            print(f"✗ Attack failed - could not flip prediction")
            self.attack_results.append({
                "attack_type": attack_type,
                "success": False,
                "original_text": text,
                "adversarial_text": text,
                "original_prediction": orig_pred,
                "adversarial_prediction": orig_pred,
                "modification": "none"
            })
            return None

    def analyze_results(self) -> pd.DataFrame:
        """analyze and summarize attack results"""
        if not self.attack_results:
            print("No attack results to analyze.")
            return None

        df = pd.DataFrame(self.attack_results)

        print("\n" + "="*70)
        print("ATTACK RESULTS SUMMARY")
        print("="*70)

        # success rate by attack type
        print("\nSuccess Rate by Attack Type:")
        success_by_type = df.groupby('attack_type')['success'].agg(['sum', 'count', 'mean'])
        success_by_type.columns = ['Successful', 'Total', 'Success Rate']
        print(success_by_type)
        print()

        return df

    def save_results(self, filename: str = "attack_results.json"):
        """save attack results to json file"""
        with open(filename, 'w') as f:
            json.dump(self.attack_results, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    """main function to run adversarial attacks"""

    print("="*70)
    print("ADVERSARIAL TEXT ATTACK DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates adversarial attacks on NLP models.")
    print("Target: Sentiment Analysis Model (DistilBERT-SST2)")
    print("Note: Using transformers 4.42.4 for M1 Mac compatibility")
    print()

    # initialize attack system
    attacker = AdversarialTextAttack(
        model_name="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # test examples - positive sentiment
    positive_examples = [
        "The movie was absolutely wonderful and I loved every minute of it!",
        "This restaurant serves the most delicious food I've ever tasted.",
        "I'm so happy with this purchase, it exceeded all my expectations!"
    ]

    # test examples - negative sentiment
    negative_examples = [
        "This movie was terrible and a complete waste of time.",
        "The service was awful and the food was cold.",
        "I absolutely hate this product, it broke after one day."
    ]

    # run attacks on positive examples (try to flip to negative)
    print("\n" + "="*70)
    print("ATTACKING POSITIVE SENTIMENT EXAMPLES (Word Substitution)")
    print("="*70)

    for i, text in enumerate(positive_examples, 1):
        print(f"\n--- Example {i} ---")
        attacker.run_comprehensive_attack(text, attack_type="word_substitution")

    # run attacks on negative examples (try to flip to positive)
    print("\n" + "="*70)
    print("ATTACKING NEGATIVE SENTIMENT EXAMPLES (Word Substitution)")
    print("="*70)

    for i, text in enumerate(negative_examples, 1):
        print(f"\n--- Example {i} ---")
        attacker.run_comprehensive_attack(text, attack_type="word_substitution")

    # character-level attacks on a few examples
    print("\n" + "="*70)
    print("CHARACTER-LEVEL ATTACKS (DeepWordBug-style)")
    print("="*70)

    for i, text in enumerate(positive_examples[:2], 1):
        print(f"\n--- Example {i} ---")
        attacker.run_comprehensive_attack(text, attack_type="character_level")

    # analyze results
    print("\n")
    df_results = attacker.analyze_results()

    # save results
    attacker.save_results("attack_results.json")

    print("\n" + "="*70)
    print("ATTACK DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nTotal attacks: {len(attacker.attack_results)}")
    print(f"Successful attacks: {sum(r['success'] for r in attacker.attack_results)}")
    print(f"Success rate: {sum(r['success'] for r in attacker.attack_results)/len(attacker.attack_results)*100:.1f}%")


if __name__ == "__main__":
    main()
