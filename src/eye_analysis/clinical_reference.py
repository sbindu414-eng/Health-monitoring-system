"""
Structured Ophthalmology Reference & Screening Report Generator (ICDR / AAO Standards).

Provides verified educational explanations and readable screening summaries strictly
grounded in the CNN's actual prediction, predictive uncertainty, and image quality metrics.
Never invents or overrides a DR grade.
"""

from typing import Dict

ICDR_REFERENCE_KB = {
    0: {
        "name": "No Apparent Diabetic Retinopathy (ICDR Grade 0)",
        "clinical_criteria": "No visible microaneurysms, retinal hemorrhages, exudates, or neovascularization.",
        "patient_explanation": (
            "The model did not find clear patterns associated with diabetic retinopathy lesions on this fundus scan. "
            "Note that a person with diabetes can still have a normal Grade 0 retina before eye complications develop."
        ),
        "recommended_action": "Continue routine annual dilated retinal screening and regular blood glucose / HbA1c monitoring.",
    },
    1: {
        "name": "Mild Non-Proliferative Diabetic Retinopathy (ICDR Grade 1)",
        "clinical_criteria": "Microaneurysms only (early focal capillary outpouchings without extensive hemorrhages or exudates).",
        "patient_explanation": (
            "The model detected subtle early retinal changes consistent with Mild NPDR. Mild NPDR often causes no vision "
            "symptoms yet, making early glycemic, blood pressure, and lipid control especially protective."
        ),
        "recommended_action": "Schedule a dilated fundus examination with an optometrist or ophthalmologist within 6–12 months.",
    },
    2: {
        "name": "Moderate Non-Proliferative Diabetic Retinopathy (ICDR Grade 2)",
        "clinical_criteria": (
            "More than microaneurysms alone (e.g., dot-and-blot hemorrhages, hard exudates, cotton-wool spots, or venous beading) "
            "but less than Severe NPDR criteria."
        ),
        "patient_explanation": (
            "The model identified patterns suggestive of Moderate NPDR (referable diabetic retinopathy). Small blood vessels in "
            "the retina may be leaking fluid or showing scattered micro-bleeds that warrant specialist examination."
        ),
        "recommended_action": "Prompt referral to an ophthalmologist for dilated retinal evaluation and optical coherence tomography (OCT) if indicated.",
    },
    3: {
        "name": "Severe Non-Proliferative Diabetic Retinopathy (ICDR Grade 3)",
        "clinical_criteria": (
            "ETDRS 4-2-1 rule: >20 intraretinal hemorrhages in all 4 quadrants, definite venous beading in >=2 quadrants, "
            "or prominent intraretinal microvascular abnormalities (IRMA) in >=1 quadrant, without neovascularization."
        ),
        "patient_explanation": (
            "The model detected strong features consistent with Severe NPDR, meaning significant areas of retinal capillary "
            "non-perfusion and vascular stress are suspected with high risk of progression."
        ),
        "recommended_action": "Urgent ophthalmology evaluation (typically within weeks) to assess need for retinal therapy.",
    },
    4: {
        "name": "Proliferative Diabetic Retinopathy (ICDR Grade 4)",
        "clinical_criteria": "Neovascularization of the disc (NVD) or elsewhere (NVE), or vitreous / preretinal hemorrhage.",
        "patient_explanation": (
            "The model identified features resembling Proliferative DR, an advanced vision-threatening stage characterized by "
            "fragile abnormal new retinal vessels or bleeding risk."
        ),
        "recommended_action": "Immediate / urgent specialist retina consultation for comprehensive examination and sight-preserving care.",
    },
}


def build_retinal_screening_report(
    predicted_grade: int,
    class_probabilities: Dict[str, float],
    confidence_pct: float,
    is_uncertain: bool,
    quality: Dict,
    any_dr_probability_pct: float,
    referable_dr_probability_pct: float,
    gradcam_info: Dict = None,
) -> Dict:
    """Compile a structured, plain-language screening narrative strictly from CNN outputs."""
    kb = ICDR_REFERENCE_KB.get(int(predicted_grade), ICDR_REFERENCE_KB[0])

    if is_uncertain:
        headline = (
            "Prediction Uncertain — Clearer Retinal Scan or Professional Review Recommended"
        )
        summary = (
            f"The CNN output carries substantial class ambiguity (top probability {confidence_pct:.1f}%, "
            f"image quality score {quality.get('quality_score', 0)}/100). "
            "Rather than forcing a confident label, clinical review or a sharper centered fundus image is recommended."
        )
    else:
        headline = kb["name"]
        summary = kb["patient_explanation"]

    sorted_probs = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top_two_text = ", ".join(f"{name}: {val:.1f}%" for name, val in sorted_probs[:2])

    limitations = [
        "Trained on public multi-center benchmark retinal datasets (EyePACS, IDRiD, DeepDRiD/RetinaMNIST, APTOS); not cleared as a standalone clinical diagnostic device.",
        "Cannot directly measure blood glucose (mg/dL) or diagnose systemic diabetes in patients without visible retinal vascular lesions.",
        "Image artifacts, media opacity (e.g., cataract), or uncalibrated mobile fundus adapters can shift class probabilities.",
    ]

    report = {
        "reference_standard": "International Clinical Diabetic Retinopathy (ICDR) Severity Scale (AAO / ETDRS)",
        "headline": headline,
        "summary": summary,
        "clinical_criteria": kb["clinical_criteria"],
        "recommended_action": kb["recommended_action"],
        "top_probabilities_summary": top_two_text,
        "any_dr_probability_pct": any_dr_probability_pct,
        "referable_dr_probability_pct": referable_dr_probability_pct,
        "quality_score": quality.get("quality_score"),
        "gradcam_focus_note": (
            f"Model attention concentrated in: {gradcam_info.get('focus_region')}."
            if gradcam_info and gradcam_info.get("available")
            else "Grad-CAM attention map unavailable for this sample."
        ),
        "limitations": limitations,
    }
    return report
