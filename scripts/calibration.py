"""Standard-library helpers. No API calls or fitted Ridge model included."""
import bisect
import csv
import math


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def option_value(stats, name):
    """stats is the content of disclosure item id='option-implied-stats'."""
    item = (stats or {}).get(name) or {}
    value = item.get('value')
    return float(value) if item.get('status') == 'ok' and finite(value) else None


def fit_option_reference(training_rows):
    """Rows: {'p': valid raw LLM percentile, 'move': decimal move or None}.

    Fit on training data only, separately for each model/prompt. Freeze for test.
    Caller must exclude failed LLM calls and unresolved/future training events.
    """
    rows = list(training_rows)
    if not rows or any(not finite(r.get('p')) or not 0 <= r['p'] <= 1 for r in rows):
        raise ValueError('Training predictions must be finite percentiles in [0,1]')
    moves = [r['move'] for r in rows if finite(r.get('move')) and r['move'] > 0]
    return {'pivot': sum(r['p'] for r in rows) / len(rows),
            'move_mean': sum(moves) / len(moves) if moves else None,
            'n_predictions': len(rows), 'n_moves': len(moves)}


def adjust_percentile(p, move, reference):
    """Official square-root shape, with frozen training reference.

    Missing/zero/invalid move or no training move: return p unchanged.
    This is not CDF(p), not return conversion, and not an additional Ridge stage.
    """
    if not finite(p) or not 0 <= p <= 1:
        raise ValueError('p must be a finite percentile in [0,1]')
    mu, mean_move = reference.get('pivot'), reference.get('move_mean')
    if not finite(mu) or not 0 <= mu <= 1:
        raise ValueError('Invalid training pivot')
    if not finite(move) or move <= 0 or not finite(mean_move) or mean_move <= 0:
        return float(p)
    return max(0., min(1., mu + (p - mu) * math.sqrt(move / mean_move)))


class HistoricalCDF:
    """Exact step ECDF F(x)=count(CAR<=x)/N; CAR is a decimal return.

    Optional conversion for an explicit CAR estimate, NOT for predicted_percentile.
    F(E[CAR]) generally differs from E[F(CAR)]; do not call it a learned calibrator.
    """
    def __init__(self, path):
        with open(path, newline='', encoding='utf-8') as handle:
            self.values = sorted(float(r['car1_decimal']) for r in csv.DictReader(handle))
        if not self.values or not all(math.isfinite(x) for x in self.values):
            raise ValueError('Empty or invalid CAR data')

    def percentile(self, car1_decimal):
        if not finite(car1_decimal):
            raise ValueError('CAR must be finite')
        return bisect.bisect_right(self.values, car1_decimal) / len(self.values)
