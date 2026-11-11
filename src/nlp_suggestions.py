"""
NLP Suggestion Engine for Indian Sign Language (ISL) Sentence Formation.
Provides context-aware n-gram and phrase-based next-word predictions.
"""

from typing import List
import re


class WordSuggestionEngine:
    """
    Context-aware n-gram and phrase-based word prediction engine.
    Predicts relevant next words based on preceding sentence context.
    """

    def __init__(self):
        # Common high-frequency default words (used when sentence is empty or no match)
        self.default_starters = [
            "I", "HELLO", "HOW", "WHAT", "MY", "THANK", "PLEASE", "GOOD", "YES", "WE"
        ]

        # Trigram dictionary: (word1, word2) -> [suggested next words]
        self.trigrams = {
            ("I", "AM"): ["HAPPY", "LEARNING", "READY", "FINE", "HERE", "STUDENT", "TIRED", "DEAF"],
            ("HOW", "ARE"): ["YOU", "THINGS", "WE", "THEY", "YOUR"],
            ("NICE", "TO"): ["MEET", "SEE", "HAVE"],
            ("TO", "MEET"): ["YOU", "ALL", "AGAIN"],
            ("WHAT", "IS"): ["YOUR", "THIS", "THAT", "THE", "TIME"],
            ("IS", "YOUR"): ["NAME", "ADDRESS", "PROBLEM", "FAMILY"],
            ("MY", "NAME"): ["IS", "WAS"],
            ("I", "WANT"): ["TO", "WATER", "FOOD", "HELP", "LEARN", "SIGN"],
            ("WANT", "TO"): ["GO", "LEARN", "EAT", "HELP", "TALK", "SLEEP"],
            ("I", "NEED"): ["HELP", "WATER", "MEDICINE", "TIME", "YOU"],
            ("THANK", "YOU"): ["VERY", "SO", "MUCH", "AGAIN", "ALL"],
            ("YOU", "VERY"): ["MUCH"],
            ("YOU", "SO"): ["MUCH"],
            ("PLEASE", "HELP"): ["ME", "US", "NOW"],
            ("GOOD", "MORNING"): ["SIR", "MADAM", "ALL", "FRIEND"],
            ("GOOD", "NIGHT"): ["ALL", "FRIEND"],
            ("SEE", "YOU"): ["LATER", "SOON", "TOMORROW", "AGAIN"],
            ("WHERE", "IS"): ["THE", "MY", "YOUR", "HOSPITAL", "BATHROOM"],
            ("CAN", "YOU"): ["HELP", "REPEAT", "SIGN", "UNDERSTAND", "COME"],
            ("I", "CAN"): ["SIGN", "HELP", "UNDERSTAND", "COME", "GO"],
            ("I", "CANNOT"): ["HEAR", "UNDERSTAND", "SPEAK", "COME"],
            ("DO", "YOU"): ["KNOW", "UNDERSTAND", "WANT", "NEED", "SIGN"],
            ("YOU", "KNOW"): ["ISL", "SIGN", "ME", "ENGLISH"],
            ("WE", "ARE"): ["HAPPY", "FRIENDS", "LEARNING", "READY"],
            ("VERY", "MUCH"): ["THANK", "HAPPY", "NICE"],
            ("HAVE", "A"): ["NICE", "GOOD", "GREAT", "SAFE"],
            ("A", "NICE"): ["DAY", "TIME", "WEEKEND"],
            ("A", "GOOD"): ["DAY", "JOB", "TIME"],
        }

        # Bigram dictionary: word1 -> [suggested next words]
        self.bigrams = {
            "I": ["AM", "WANT", "NEED", "LIKE", "CAN", "KNOW", "GO", "SEE", "HAVE"],
            "AM": ["HAPPY", "FINE", "LEARNING", "READY", "HERE", "TIRED", "GOOD"],
            "HOW": ["ARE", "DO", "CAN", "MUCH", "MANY", "IS"],
            "ARE": ["YOU", "WE", "THEY", "ALL", "READY"],
            "YOU": ["ARE", "CAN", "KNOW", "WANT", "WELCOME", "LATER", "TOO"],
            "WHAT": ["IS", "ARE", "DO", "CAN", "HAPPENED", "YOUR"],
            "MY": ["NAME", "FAMILY", "FRIEND", "HOUSE", "FATHER", "MOTHER", "TEACHER"],
            "YOUR": ["NAME", "FAMILY", "HELP", "FRIEND", "HOME"],
            "NAME": ["IS", "WAS"],
            "THANK": ["YOU", "VERY"],
            "PLEASE": ["HELP", "GIVE", "WAIT", "COME", "REPEAT", "SIGN"],
            "GOOD": ["MORNING", "AFTERNOON", "EVENING", "NIGHT", "JOB", "LUCK", "DAY"],
            "WE": ["ARE", "CAN", "WANT", "LEARN", "GO", "NEED"],
            "WHERE": ["IS", "ARE", "DO", "CAN"],
            "WHO": ["IS", "ARE", "CAN"],
            "WHEN": ["CAN", "WILL", "DO", "IS"],
            "WHY": ["ARE", "DID", "DO", "NOT"],
            "CAN": ["YOU", "I", "WE"],
            "WANT": ["TO", "WATER", "FOOD", "HELP"],
            "NEED": ["HELP", "WATER", "FOOD", "DOCTOR"],
            "NICE": ["TO", "DAY"],
            "SEE": ["YOU", "AGAIN", "LATER"],
            "YES": ["I", "WE", "PLEASE", "THANK", "SURE"],
            "NO": ["PROBLEM", "THANK", "I", "WORRY"],
            "SIGN": ["LANGUAGE", "ISL", "AGAIN", "SLOWLY"],
            "LANGUAGE": ["ISL", "INDIAN", "GOOD"],
            "INDIAN": ["SIGN", "LANGUAGE"],
            "TIME": ["IS", "NOW", "OVER"],
            "WATER": ["PLEASE", "NEED"],
            "FOOD": ["PLEASE", "HUNGRY", "WANT"],
            "HELP": ["ME", "PLEASE", "NEEDED"],
            "AGAIN": ["PLEASE", "THANK"],
            "HELLO": ["HOW", "MY", "I", "FRIEND", "EVERYONE"],
            "HI": ["HOW", "MY", "I", "FRIEND"],
        }

    def clean_text(self, text: str) -> List[str]:
        """Tokenize and uppercase clean words."""
        if not text:
            return []
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', text.upper())
        return [w for w in cleaned.split() if w]

    def get_suggestions(self, sentence: str, max_count: int = 5) -> List[str]:
        """
        Return up to `max_count` relevant, non-duplicate next word suggestions.
        Follows hierarchical resolution: Trigram -> Bigram -> Default starters.
        """
        words = self.clean_text(sentence)
        suggestions: List[str] = []

        # If sentence has 2+ words, check trigrams first
        if len(words) >= 2:
            prev_two = (words[-2], words[-1])
            if prev_two in self.trigrams:
                for w in self.trigrams[prev_two]:
                    if w not in suggestions and w != words[-1]:
                        suggestions.append(w)

        # If we need more, check bigrams on the last word
        if len(suggestions) < max_count and len(words) >= 1:
            last_word = words[-1]
            if last_word in self.bigrams:
                for w in self.bigrams[last_word]:
                    if w not in suggestions and w != last_word:
                        suggestions.append(w)

        # If still need more, use context-aware default high-frequency words
        if len(suggestions) < max_count:
            for w in self.default_starters:
                if w not in suggestions and (len(words) == 0 or w != words[-1]):
                    suggestions.append(w)
                    if len(suggestions) >= max_count:
                        break

        return suggestions[:max_count]
