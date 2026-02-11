def calculate_score(key_answers, student_answers):
    total = len(key_answers)
    if total == 0:
        return 0
    correct = sum(1 for q in key_answers if key_answers[q] == student_answers.get(q, None))
    return round((correct / total) * 100, 2)
