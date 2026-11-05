"""Single-opponent performance scores using FIDE table 8.1.1."""
from decimal import Decimal, ROUND_HALF_UP

TABLE_SOURCE = 'https://handbook.fide.com/chapter/B022024'
# dp for percentage scores 50..100; the negative half is symmetric.
POSITIVE_DP = (0,7,14,21,29,36,43,50,57,65,72,80,87,95,102,110,117,125,
               133,141,149,158,166,175,184,193,202,211,220,230,240,251,262,
               273,284,296,309,322,336,351,366,383,401,422,444,470,501,538,589,677,800)


def performance(wins, draws, losses, teacher=1800):
    if any(type(n) is not int or n < 0 for n in (wins, draws, losses)):
        raise ValueError('W/D/L must be nonnegative integer counts')
    total = wins + draws + losses
    if not total:
        return None
    score = Decimal(2*wins+draws)/Decimal(2*total)
    percent = int((100*score).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    dp = POSITIVE_DP[abs(percent-50)] * (1 if percent >= 50 else -1)
    return {'games':total, 'score_rate':float(score), 'table_percent':percent,
            'rating_difference':dp, 'performance_rating':teacher+dp,
            'teacher_rating':teacher, 'method':'fide-table-8.1.1',
            'rounding':'nearest whole percentage, half up',
            'nominal_endpoint':percent in (0,100)}
