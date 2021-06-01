import seaborn as sns
from method import Method

palette = sns.color_palette()

COLORS = {method.value: color for method, color in zip(Method, palette)}

LABELS = {
    Method.SPICE: 'SPICE',
    Method.CREPE_TINY: 'CREPE tiny',
    Method.DDSP_INV: 'DDSP-INV',
    Method.YIN: 'YIN',
}

THRESHOLDS = {
    Method.SPICE: 0.88,
    Method.CREPE_TINY: 0.67,
    Method.DDSP_INV: 0.5,
    Method.YIN: 0.5,
}
