from transformers import pipeline

qa_pipeline = None

def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
            model_name = "distilbert-base-cased-distilled-squad"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Warning: Could not initialize HF pipeline: {e}")
            qa_pipeline = False
    return qa_pipeline if qa_pipeline is not False else None



class NLPService:
    @staticmethod
    def answer_question(question: str) -> str:
        """
        Answers a regulatory question using a pre-trained NLP model.
        In a real system, the 'context' would be pulled from a database of
        regulatory documents. For this demo, we'll use a hard-coded context.
        """
        context = """
        The Markets in Crypto-Assets (MiCA) regulation, effective June 2024,
        requires all Crypto-Asset Service Providers (CASPs) operating within the EU
        to report any transaction exceeding EUR 10,000 to the relevant national
        competent authority within 24 hours. Furthermore, all cross-border
        crypto-asset transfers must include information about their originator
        and beneficiary, a rule commonly known as the 'Travel Rule'.
        For internal fraud monitoring, firms must implement systems capable of
        detecting and flagging suspicious insider trading patterns, particularly
        around the announcement of new asset listings.
        """
        
        try:
            pipeline_instance = get_qa_pipeline()
            if pipeline_instance:
                result = pipeline_instance(question=question, context=context)
                if result.get('score', 0) >= 0.1:
                    return result['answer']

            # Fallback answer extraction based on context sentences if HF pipeline is unavailable/failed
            question_words = set(question.lower().split())
            sentences = [s.strip() for s in context.strip().split('.') if s.strip()]
            best_sentence = ""
            max_overlap = 0

            for sentence in sentences:
                overlap = len(question_words.intersection(set(sentence.lower().split())))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence

            if best_sentence and max_overlap > 1:
                return best_sentence + "."
            
            return "According to MiCA regulations, CASPs must report transactions over EUR 10,000 within 24 hours and comply with the Travel Rule."
        except Exception as e:
            return f"Error processing question: {str(e)}"