from decimal import localcontext

import pytest

from memorybot.templates import TemplateRegistry


@pytest.mark.parametrize("reverse", [False, True])
def test_csv_summary_preserves_small_values_through_cancellation(reverse):
    rows = ["A,10000000000000000000000000000", "A,1", "A,-10000000000000000000000000000"]
    if reverse:
        rows.reverse()
    text = "Group,Value\n" + "\n".join(rows)
    with localcontext(prec=4):
        result = TemplateRegistry().execute("csv_summary", "", [{"text": text}])
    assert result["total"] == "1"
    assert result["csv"] == "Group,Value\nA,1\n总计,1\n"


def test_csv_summary_keeps_exact_totals_across_groups_and_documents():
    documents = [{"text": "分组,金额\n甲,1e80\n乙,0.125\n"}, {"text": "分组,金额\n丙,-1e80\n"}]
    result = TemplateRegistry().execute("csv_summary", "", documents)
    assert result["total"] == "0.125"
    assert result["csv"].endswith("总计,0.125\n")
