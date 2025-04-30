# adds forward hooks to models and computes internal loss components between residuals of teacher and student


class InternalObjective:
    def serialize(self):
        return {
            "name": self.__class__.__name__,
            "type": "internal",
        }