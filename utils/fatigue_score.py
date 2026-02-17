class FatigueScorer:
    def __init__(self, weights):
        """
        Args:
            weights: dictionary containing weights for different factors.
                     {'eye': 0.4, 'yawn': 0.3, 'head': 0.2, 'blink': 0.1}
        """
        self.weights = weights
        self.score = 0
        
    def calculate_score(self, eye_closed_duration, yawn_frequency, head_tilt_angle, blink_rate_deviation):
        """
        Calculates the fatigue score (0-100).
        All inputs should be normalized to 0-1 range where 1 is worst case.
        """
        
        # Weighted sum
        raw_score = (
            (eye_closed_duration * self.weights['eye']) + 
            (yawn_frequency * self.weights['yawn']) + 
            (head_tilt_angle * self.weights['head']) + 
            (blink_rate_deviation * self.weights['blink'])
        )
        
        # Scale to 0-100
        self.score = min(100, max(0, raw_score * 100))
        return self.score
