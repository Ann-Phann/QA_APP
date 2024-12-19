class OutputFormatter:
    @staticmethod
    def get_output_format(question_type: str) -> str:
        formats = {
            "Yes/No" : "Start with a clear 'Yes' or 'No' followed by a brief explanation.",
            "Explanation" : "Provide a clear and concise explanation.",
            "List" : "Provide your response as a numbered list of points.",
            "Comparison" : "Structure your response with clear comparisons with main differences.",
            "Other" : "Provide a clear, concise response."
        }
        return formats.get(question_type, formats["Other"])
    
    @staticmethod
    def get_reasoning_examples(question_type: str) -> str:
        examples = {
            "Yes/No" : [
                {"question": "Is water wet?", "response": "Yes. Water is wet because it adheres to surfaces and creates the sensation of wetness."},
                {"question": "Is fire cold?", "response": "No. Fire produces heat and is not cold."}
            ],
            "Explanation" : [{"question": "Why is the sky blue?", "response": "The sky appears blue due to the scattering of sunlight by the Earth's atmosphere."
                              }], 
            "List" : [
                {"question": "What are the primary colors?", "response": "The primary colors are red, blue, and yellow."},
                {"question": "List the planets in the solar system.", "response": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."}
            ],
            "Comparison" : [
                {"question": "What is the difference between a lion and a tiger?", "response": "Lions are social and live in groups called prides, while tigers are solitary and have striped coats."},
                {"question": "Compare a desktop computer and a laptop.", "response": "A desktop computer is stationary and more powerful, while a laptop is portable and compact."}
            ]
        }
            
        
        return examples.get(question_type, [])
