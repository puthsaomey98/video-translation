import re
from deep_translator import GoogleTranslator

class KhmerSpokenNaturalizer:
    def __init__(self):
        # 🌟 Advanced Colloquial Phrase Mapping (Stiff/Literal MT ➔ Natural Drama Spoken Khmer)
        self.colloquial_phrases = {
            "តើអ្នកជាใคร": "អ្នកជាអ្នកណា?",
            "តើ​អ្នក​ជា​នរណា": "ឯងជាអ្នកណា?",
            "តើคุณเป็นใคร": "ឯងជាអ្នកណា?",
            "ចាកចេញទៅ": "ទៅឲ្យផុតទៅ!",
            "ទៅបាត់": "ទៅឲ្យផុតទៅ!",
            "មានរឿងអីកើតឡើង": "កើតអីហ្នឹង?",
            "មានអ្វីកើតឡើង": "មានរឿងអីហ្នឹង?",
            "តើមានរឿងអ្វីកើតឡើង": "កើតរឿងអីឡើងហ្នឹង?",
            "ខ្ញុំមិនដឹងទេ": "អត់ដឹងទេ!",
            "ខ្ញុំមិនដឹង": "អត់ដឹងទេ!",
            "មិនបាច់ទេ": "មិនបាច់ទេអី!",
            "អរគុណច្រើន": "អរគុណច្រើនណាស់!",
            "សូមទោស": "សុំទោស!",
            "តោះទៅ": "តោះទៅ!",
            "ចាំបន្តិច": "ចាំបន្តិច!",
            "ឯងចង់មានរឿងមែនទេ": "ចង់មានរឿងជាមួយអញមែនទេ?",
            "អ្នកចង់មានរឿងមែនទេ": "ចង់មានរឿងមែនទេ?",
            "សុំអង្គុយទីនេះផងបានទេ": "សុំអង្គុយទីនេះមួយភ្លែតបានទេ?",
            "ឯងមិនទាន់ភ្លេចខ្ញុំទេមែនទេ": "ឯងមិនទាន់ភ្លេចខ្ញុំទេហេស?",
            "កៅអីទំនេរពេញហ្នឹង": "កៅអីទំនេរពេញហ្នឹង ម៉េចមិនទៅអង្គុយ?",
        }

        # Conversational Regex Replacement Rules
        self.regex_rules = [
            # Remove crude or harsh pronouns if unwanted, map to natural spoken ones
            (r'\bអញ\b', 'ខ្ញុំ'),
            
            # Remove stiff written passive voice markers
            (r'ត្រូវបាន\s*', ''),
            
            # Remove unnecessary written possessive 'នៃ'
            (r'\s+នៃ\s+', ' '),
            
            # Replace formal written conjunctions & adverbs with spoken ones
            (r'មិនមែនទេ', 'អត់ទេ'),
            (r'យ៉ាងណាក៏ដោយ', 'ប៉ុន្តែ'),
            (r'លើសពីនេះទៅទៀត', 'ហើយ'),
            (r'ជាមួយគ្នានេះដែរ', 'ហើយម្យ៉ាងទៀត'),
            (r'រូបលោក', 'គាត់'),
            (r'លោកអ្នក', 'អ្នក'),
            (r'ដោយសារតែ', 'ព្រោះ'),
            (r'ប្រហែលជា', 'ប្រហែល'),
            
            # Clean up formal question structures ("តើ ... ឬទេ?" -> "... មែនទេ?")
            (r'តើ\s+(.*?)\s+ឬទេ\?', r'\1 មែនទេ?'),
            (r'តើ\s+', ''),
            
            # Clean up formal exclamation endings
            (r'យ៉ាងជាក់ច្បាស់', 'ណាស់'),
        ]

    def clean_noise(self, text: str) -> str:
        """Strips subtitle noise tags like [Music], (Applause), <i>, etc."""
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def naturalize(self, text: str) -> str:
        """Converts raw translated Khmer into fluent conversational drama language."""
        if not text:
            return ""
            
        text = self.clean_noise(text)
        
        # 1. Check direct colloquial phrase matching
        for formal_phrase, spoken_phrase in self.colloquial_phrases.items():
            if formal_phrase in text:
                text = text.replace(formal_phrase, spoken_phrase)

        # 2. Apply general spoken grammar & structural cleanup rules
        for pattern, replacement in self.regex_rules:
            text = re.sub(pattern, replacement, text)
            
        # Clean up extra spaces
        return re.sub(r'\s+', ' ', text).strip()


class KhmerTranslator:
    def __init__(self):
        print("Initializing Advanced Spoken Khmer Translator & Naturalizer...")
        self.naturalizer = KhmerSpokenNaturalizer()

    def translate_text(self, text: str, source_lang: str = "en") -> str:
        """Translates text string from source language to natural spoken Khmer."""
        try:
            src = source_lang if source_lang else "en"
            
            lang_mapping = {
                "zh": "zh-CN",
                "zh-cn": "zh-CN",
                "zh-tw": "zh-TW"
            }
            resolved_src = lang_mapping.get(src.lower(), src)

            # Step 1: Raw translation via Google Translate
            translator = GoogleTranslator(source=resolved_src, target="km")
            translated_text = translator.translate(text)

            if not translated_text:
                return text

            # Step 2: Post-process into Spoken Drama Language (ភាសានិយាយ)
            spoken_khmer = self.naturalizer.naturalize(translated_text)
            
            return spoken_khmer if spoken_khmer else translated_text

        except Exception as e:
            print(f"❌ Translation error ({source_lang}): {str(e)}")
            return text