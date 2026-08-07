from deep_translator import GoogleTranslator

def translate_english_to_khmer(text: str) -> str:
    """Translates an English string into Khmer using Google Translate via deep-translator."""
    try:
        print(f"Original English: {text}")
        # Initialize GoogleTranslator for English ('en') to Khmer ('km')
        translator = GoogleTranslator(source='en', target='km')
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return ""

if __name__ == "__main__":
    # Sample text to test
    sample_text = "All right, so here we are, in front of the elephants."
    
    print("Initializing translation...")
    khmer_result = translate_english_to_khmer(sample_text)
    
    print("-" * 40)
    print(f"Translated Khmer: {khmer_result}")
    print("-" * 40)
    
    # Optional: If you want to save the result to a text file
    output_filename = "khmer_translation_output.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(khmer_result)
    print(f"✅ Khmer text successfully saved to: {output_filename}")