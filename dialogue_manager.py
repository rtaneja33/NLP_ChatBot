### ROHAN T. - Stack Overflow Chat Bot
import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from chatterbot.trainers import ChatterBotCorpusTrainer

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
 

        question_vec = question_to_vec(
            question, self.word_embeddings, self.embeddings_dim).reshape(-1, self.embeddings_dim)

        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings, metric="cosine")[0]

        return thread_ids[best_thread]
        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
#         question_vec = question_to_vec(question, thread_embeddings, self.embeddings_dim)
#         best_thread = pairwise_distances_argmin(
#             X = question_vec.reshape(-1, self.embeddings_dim), Y=thread_embeddings, metric='cosine'
#         )[0]
        
#         return thread_ids[best_thread]



    
class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
#         self.chitchat_bot = ChatBot(
#             'RohanBot', trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
#         self.chitchat_bot.train("chatterbot.corpus.english")
        
        self.chitchat_bot = ChatBot("mybot",
        logic_adapters=[
        {
            'import_path': 'chatterbot.logic.SpecificResponseAdapter',
            'input_text': 'What is your name',
            'output_text': 'My name is NLPBot for now... my creator is Rohan'
        },
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
            "response_selection_method": "chatterbot.response_selection.get_first_response",
            'maximum_similarity_threshold': 0.96
        },
        {
            'import_path': 'chatterbot.logic.LowConfidenceAdapter',
            'threshold': 0.45,
            'default_response': 'I am sorry but I am not that smart yet.. I do not quite understand.'
        } ], trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
        
        self.chitchat_bot.train("chatterbot.corpus.english")

  
     

        ########################
        #### YOUR CODE HERE ####
        ########################
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag[0])
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

