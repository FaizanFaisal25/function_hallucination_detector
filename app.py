import ast
import re
import streamlit as st
from typing import List, Tuple, Optional
from collections import Counter

st.set_page_config(page_title="LLM Code Hallucination Detector", layout="wide")


def extract_signature_with_ast(code: str) -> Optional[Tuple[str, List[str]]]:
    """Try to parse a Python function signature using ast. Returns (name, [param_names]) or None."""
    code = code.strip()
    if not code:
        return None

    # If user provided just the def line without a body, append a pass so ast can parse.
    if re.match(r"^def\s+\w+\s*\([^)]*\)\s*:$", code.splitlines()[0].strip()):
        code = code + "\n    pass\n"

    try:
        node = ast.parse(code)
    except Exception:
        return None

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            name = item.name
            params = []
            for arg in item.args.args:
                params.append(arg.arg)
            # handle *args / **kwargs
            if item.args.vararg:
                params.append("*" + item.args.vararg.arg)
            if item.args.kwarg:
                params.append("**" + item.args.kwarg.arg)
            return name, params
    return None


def extract_signature_with_regex(code: str) -> Optional[Tuple[str, List[str]]]:
    """Fallback regex-based signature extraction. Supports basic cases."""
    m = re.search(r"def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)", code)
    if not m:
        return None
    name = m.group("name")
    params_raw = m.group("params").strip()
    if not params_raw:
        return name, []
    parts = [p.strip() for p in params_raw.split(',') if p.strip()]
    names = []
    for p in parts:
        # remove defaults / type annotations
        p = re.sub(r"=.*", "", p)
        p = p.split(':')[0].strip()
        names.append(p)
    return name, names


def parse_signature(code: str) -> Tuple[Optional[str], List[str]]:
    """Return (function_name_or_None, [param names]) from either AST or regex approach."""
    res = extract_signature_with_ast(code)
    if res:
        return res[0], res[1]
    res = extract_signature_with_regex(code)
    if res:
        return res[0], res[1]
    return None, []


# ------------------------------ NEW comparison logic ------------------------------
def compare_signatures(
    gt_name: Optional[str], 
    gt_params: List[str], 
    pred_name: Optional[str], 
    pred_params: List[str]
):
    """
    Return a structured report of mismatches, handling:
      - function name mismatch
      - parameter OK vs. reordered vs. missing vs. extra vs. wrong_name
    in a more robust multi-step way.
    """
    report = {
        "function_name_mismatch": False,
        "function_name_gt": gt_name,
        "function_name_pred": pred_name,
        "params": [],
        "summary": {"wrong_name": 0, "missing": 0, "extra": 0}
    }

    # 1) Check function name
    if gt_name is not None and pred_name is not None and gt_name != pred_name:
        report["function_name_mismatch"] = True

    # We will collect our results in a list of dicts in the order of the ground-truth params,
    # then append any "extra" from the prediction that remain unmatched.
    results = []

    # Keep track of which predicted indices are already "used" by a match.
    used_pred_indices = set()

    # STEP 1: Mark direct "OK" matches (same index, same parameter).
    min_len = min(len(gt_params), len(pred_params))
    for i in range(min_len):
        if gt_params[i] == pred_params[i]:
            # Perfect match at position i
            results.append({
                "type": "ok",
                "gt": gt_params[i],
                "pred": pred_params[i],
                "pos": i
            })
            used_pred_indices.add(i)
        else:
            # We'll handle mismatches later
            results.append(None)  # placeholder

    # If ground-truth has more params than prediction, fill with None placeholders.
    for i in range(min_len, len(gt_params)):
        results.append(None)

    # STEP 2: For any ground-truth param that is not yet matched (results[i] is None),
    # decide if it is "reordered" or truly "missing."
    for i in range(len(gt_params)):
        if results[i] is not None:
            continue
        gt_param = gt_params[i]
        # Try to find this same param in pred_params, at an unmatched index
        if gt_param in pred_params:
            pred_index = pred_params.index(gt_param)
            # If that index is already used, try to find a different occurrence
            # in case it appears multiple times in pred_params.
            start_search = 0
            while pred_index in used_pred_indices:
                try:
                    pred_index = pred_params.index(gt_param, start_search)
                    start_search = pred_index + 1
                except ValueError:
                    pred_index = -1
                    break

            if pred_index >= 0 and pred_index not in used_pred_indices:
                # Found the ground-truth param in prediction, at a different position => "reordered"
                results[i] = {
                    "type": "reordered",
                    "gt": gt_param,
                    "pred": gt_param,
                    "pos": i,
                    "pred_pos": pred_index
                }
                used_pred_indices.add(pred_index)
            else:
                # The param is in pred_params but positions are all used => treat as missing
                results[i] = {
                    "type": "missing",
                    "gt": gt_param,
                    "pred": None,
                    "pos": i
                }
                report["summary"]["missing"] += 1

        else:
            # The ground-truth param is not at all in the prediction => "missing"
            results[i] = {
                "type": "missing",
                "gt": gt_param,
                "pred": None,
                "pos": i
            }
            report["summary"]["missing"] += 1

    # STEP 3: Now label "extra" for any predicted parameters that are not used yet.
    # Also decide if "wrong_name" should appear (if a predicted param isn't in the ground truth at all).
    for j in range(len(pred_params)):
        if j not in used_pred_indices:
            pred_param = pred_params[j]
            if pred_param in gt_params:
                # The ground-truth does have this name, but apparently all occurrences
                # of it are used up or the positions didn't line up => call it "reordered" or "extra."
                #
                # Usually you'd call this "reordered," but we can also call it "extra" if
                # we want a single-liner.  To match the original style, let's call it "extra"
                # because there's no unmatched GT param left wanting that name.
                results.append({
                    "type": "extra",
                    "gt": None,
                    "pred": pred_param,
                    "pos": j
                })
                report["summary"]["extra"] += 1
            else:
                # This predicted param is not in ground truth at all => "wrong_name" or "extra"?
                # The original code used "wrong_name" if the param is nowhere in GT.
                # We'll do that to keep consistency with the old summary counters.
                results.append({
                    "type": "wrong_name",
                    "gt": None,
                    "pred": pred_param,
                    "pos": j
                })
                report["summary"]["wrong_name"] += 1

    # Collate all results
    report["params"] = [r for r in results if r is not None]

    return report


# Color legend
COLOR_FN_MISMATCH = "#ff4d4d"       # red
COLOR_PARAM_WRONG_NAME = "#ffcc66"  # orange
COLOR_PARAM_MISSING = "#c299ff"     # purple
COLOR_PARAM_EXTRA = "#66b3ff"       # blue
COLOR_OK = "#074707"               # green
COLOR_REORDER = "#074707"          # green (you could choose a different shade if you prefer)

st.title("LLM Code Hallucination Detector — Function Signature Checker")
st.write(
    "Provide the ground-truth function signature/code on the left "
    "and the LLM's predicted function on the right. The app will "
    "detect mismatches in function name, parameter names, extra and "
    "missing parameters, and highlight them."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ground truth (expected)")
    gt_code = st.text_area(
        "Ground truth code or signature",
        value="def sum_numbers(i, k, s):\n    pass",
        height=180,
        key="gt",
    )
    st.markdown("---")
    st.write(
        "Tip: You can paste a full function or just the `def` line. "
        "Defaults and type annotations are supported."
    )

with col2:
    st.subheader("LLM prediction")
    pred_code = st.text_area(
        "LLM predicted code or signature",
        value="def sum_numbers(s, k):\n    pass",
        height=180,
        key="pred",
    )

if st.button("Analyze"):
    gt_name, gt_params = parse_signature(gt_code)
    pred_name, pred_params = parse_signature(pred_code)

    if gt_name is None:
        st.error(
            "Could not parse a function name from the ground truth input. "
            "Make sure it has a valid `def ...(...)`."
        )
    if pred_name is None:
        st.error(
            "Could not parse a function name from the prediction input. "
            "Make sure it has a valid `def ...(...)`."
        )

    if gt_name is not None and pred_name is not None:
        report = compare_signatures(gt_name, gt_params, pred_name, pred_params)

        # Summary
        st.header("Detection Results")
        left, right = st.columns([1, 2])
        with left:
            fn_status = "OK" if not report["function_name_mismatch"] else "MISMATCH"
            fn_color = (
                "black" if not report["function_name_mismatch"] else COLOR_FN_MISMATCH
            )
            st.markdown(
                f"**Function name**: <span style='background:{fn_color};"
                f"padding:4px;border-radius:4px'>{fn_status}</span>",
                unsafe_allow_html=True,
            )
            st.write(f"Ground truth: `{report['function_name_gt']}`")
            st.write(f"Prediction: `{report['function_name_pred']}`")

            st.markdown("**Parameter summary**")
            st.write(
                f"Wrong-name: {report['summary']['wrong_name']}, "
                f"Missing: {report['summary']['missing']}"
                # f"Extra: {report['summary']['extra']}"
            )

        with right:
            st.subheader("Detailed parameter comparison")
            rows_html = []
            rows_html.append("<table style='width:100%;border-collapse:collapse'>")
            rows_html.append(
                "<tr>"
                "<th style='text-align:left;padding:6px'>Position</th>"
                "<th style='text-align:left;padding:6px'>Ground truth</th>"
                "<th style='text-align:left;padding:6px'>Prediction</th>"
                "<th style='text-align:left;padding:6px'>Status</th>"
                "</tr>"
            )

            for p in report["params"]:
                pos = p.get("pos")
                gt = p.get("gt")
                pr = p.get("pred")
                t = p.get("type")

                if t == "ok":
                    bg = COLOR_OK
                    status = "OK"
                elif t == "wrong_name":
                    bg = COLOR_PARAM_WRONG_NAME
                    status = "Wrong name"
                elif t == "missing":
                    bg = COLOR_PARAM_MISSING
                    status = "Missing in prediction"
                elif t == "extra":
                    bg = COLOR_PARAM_EXTRA
                    status = "Extra parameter (only in prediction)"
                elif t == "reordered":
                    bg = COLOR_REORDER
                    status = "Reordered / present at different position"
                else:
                    bg = "white"
                    status = t

                # Position might be None if it's one of the leftover reorder/missing/extra
                pos_str = str(pos) if pos is not None else "-"
                rows_html.append(
                    f"<tr style='background:{bg};'>"
                    f"<td style='padding:6px'>{pos_str}</td>"
                    f"<td style='padding:6px'>{gt}</td>"
                    f"<td style='padding:6px'>{pr}</td>"
                    f"<td style='padding:6px'>{status}</td>"
                    f"</tr>"
                )

            rows_html.append("</table>")
            st.markdown("\n".join(rows_html), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Legend")
        st.markdown(
            f"<div style='display:flex;gap:12px'>"
            f"<div style='background:{COLOR_FN_MISMATCH};padding:6px;border-radius:4px'>Function name mismatch</div>"
            f"<div style='background:{COLOR_PARAM_WRONG_NAME};padding:6px;border-radius:4px'>Wrong Name / Extra Parameter</div>"
            f"<div style='background:{COLOR_PARAM_MISSING};padding:6px;border-radius:4px'>Missing parameter</div>"
            # f"<div style='background:{COLOR_PARAM_EXTRA};padding:6px;border-radius:4px'>Wrong Name / Extra parameter</div>"
            f"<div style='background:{COLOR_OK};padding:6px;border-radius:4px'>OK</div>"
            f"<div style='background:{COLOR_REORDER};padding:6px;border-radius:4px'>Reordered</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("\n---\n")
        st.subheader("Raw parsed signatures")
        st.write({"ground_truth_name": gt_name, "ground_truth_params": gt_params})
        st.write({"pred_name": pred_name, "pred_params": pred_params})

st.markdown("---")
st.caption(
    "Colors: red=function name mismatch · orange=parameter name mismatch · "
    "purple=missing param · blue=extra param · green=ok or reordered"
)

# End of app
