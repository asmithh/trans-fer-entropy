class TopicModelScorer(object):
    """
    Abstract class that provides functionality to load topic information and score documents based on topic prevalence.
    DO NOT USE THIS BASE CLASS; make a class that inherits from this class instead.
    None of the functions are properly implemented but should serve as a template for specific implementations.
    """
    def __init__(self, topic_names=[], topic_data=None):
        """
        Initialize a TopicModelScorer object.

        Inputs:
            topic_names: names of the topics that are being scored by this object
            topic_data: object that contains the data necessary for computing topic scores.

        NOTE: topic_names should be the same length as topic_data
        """
        assert len(topic_names) == len(topic_data)
        self.topic_names = topic_names
        self.topic_data = topic_data

    def _score_one_topic(self, document, topic, indiv_data):
        """
        Given a document, a topic, and its data, assign a score for that topic to the document.

        Inputs:
            document: string; assumed to be bag-of-words. Implement preprocessing as necessary.
            topic: the name of the topic being scored
            indiv_data: the data associated with the topic being scored

        Outputs:
            score: float scoring topic prevalence in the input document
        """
        return 1.0

    def score_document_all_topics(self, document):
        """
        Given a document, return scores for all topics.

        Inputs:
            document: a string; assumed to be bag-of-words. Implement preprocessing as necessary.

        Ouputs:
            scores: a dict mapping topic names to scores for the document
        """
        scores = {}

        for name, indiv_data in zip(self.topic_names, self.topic_data):
            scores[name] = self._score_one_topic(document, name, indiv_data)    

        return scores

