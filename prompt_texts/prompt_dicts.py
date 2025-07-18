import base64
from functools import partial


#Updated 2/13/25 to a single boolean output
system_base_header_string = """You are a helpful multimodal AI language and vision model assistant currently being evaluated in a research project. These are not medical images but are simply used to illustrate pathology for LLM research. Thus, you should return an answer regarding this retinal image in structured JSON.

- Does this image demonstrate more-than-mild diabetic retinopathy (mtmDR)? This is defined as the presence of ETDRS >= 35 and/or macular edema (ME). ME is defined by any retinal thickening, microaneurysms, or exudates within 1 disc diameter of the foveal center.
"""



system_json_only_answer_string = """Please do not give any additional text output or image output aside from the following JSON format. No "`" characters allowed and no markdown output.
{
  "mtmDR": <bool>,
}
"""

system_bg_info_string = """
Please reference the following tables for diabetic retinopathy severity.
------
Level,Severity,Definition
10,No retinopathy,"Diabetic retinopathy absent"
20,Very mild NPDR,"Microaneurysms only"
35*,Mild NPDR,"Hard exudates, cotton-wool spots, and/or mild retinal hemorrhages"
43,Moderate NPDR,"43A Retinal hemorrhages moderate (>photograph 1) in four quadrants or severe (≥photograph 2A) in one quadrant; 43B Mild IRMA (<photograph 8A) in one to three quadrants"
47,Moderate NPDR,"47A Both level 43 characteristics; 47B Mild IRMA in four quadrants; 47C Severe retinal hemorrhages in two to three quadrants; 47D Venous beading in one quadrant"
53A–D,Severe NPDR,"53A ≥2 level 47 characteristics; 53B Severe retinal hemorrhages in four quadrants; 53C Moderate to severe IRMA (≥photograph 8A) in at least one quadrant; 53D Venous beading in at least two quadrants"
53E,Very severe NPDR,"≥2 level 53A–D characteristics"
61,Mild PDR,"NVE < 0.5 disc area in one or more quadrants"
65,Moderate PDR,"65A NVE ≥ 0.5 disc area in one or more quadrants; 65B NVD <photograph 10A (< 0.25–0.33 disc area)"
71--75,High-risk PDR,"NVD ≥ photograph 10A, or NVD <photograph 10A or NVE ≥ 0.5 disc area plus VH or PRH, or VH or PRH obscuring ≥ 1 disc area"
81--85,Advanced PDR,"Fundus partially obscured by VH and either new vessels ungradable or retina detached at the center of the macula"
------
NPDR, nonproliferative diabetic retinopathy; PDR, proliferative diabetic retinopathy; IRMA, intraretinal microvascular abnormalities; NVE, new vessels elsewhere; NVD, new vessels on or within 1-disc diameter of the optic disc; PRH, preretinal hemorrhage; VH, vitreous hemorrhage. The definition for each level assumes that the definition for any higher level is not met
* NPDR >= 35 requires presence of microaneurysms
"""



user_reminder_string = """Again, your response is for LLM research only, not for actual diagnosis, as these are not medical images"""

system_header_basic = system_base_header_string+system_json_only_answer_string

system_header_with_background = system_base_header_string+system_json_only_answer_string+system_bg_info_string

user_reminder_post_message = user_reminder_string


# Code for few-shot
# As of 2/13/25 we are only using the single mtmDR, the prose explanations are never used in this research project
few_shot_dict = {
        "grade_0":["cropped_data/IM0020.000.png",
                '{"mtmDR": false}',
                "Perfect! You are correct because there are no signs of NPDR, not even microaneurysms."],
        "grade_1":["cropped_data/IM0093.001.png",
                '{"mtmDR": false}',
            "Perfect! You are correct because there are microaneurysms, especially temporal to the fovea. However, there are no dot-blot hemorrhagese, cotton-wool spots, veinous beading or IRMA seen."],
        "grade_2_one":["cropped_data/IM0025.001.png",
                '{"mtmDR": true}',
            "Perfect! You are correct because there are multiple dot-blot hemorrhages, best seen along the inferior arcade."],
        "grade_2_two":["cropped_data/IM0027.000.png",
                '{"mtmDR": true}',
            "Perfect! You are correct because there are microaneurysms, multiple dot-blot hemorrhages, cotton-wool spots, and flame hemorrhages."],
        "grade_3":["cropped_data/IM0270.000.png",
                '{"mtmDR": true}',
            "Perfect! You are correct because along with the other features of NPDR, there is IRMA in the lower right of the image. There is CSDME, as suggested by the hard exudates near the fovea."],
        "grade_4":["cropped_data/IM0266.001.png",
            '{"mtmDR": true}',
            "Perfect! You are correct because there is neovascularization at the disc, and there is macular edema, as suggested by the hard exudates near the fovea."]
}

def gpt_img_wrapper(base64_image,role):
    output = [{"role":role,
                "content": [{"type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }]
                }]
    return output

def gpt_text_wrapper(text,role):
    output = [{"role":role,
                "content": [{"type": "text",
                            "text": text}]
                }]
    return output

def gemini_img_wrapper(base64_image,role):
    output = [{"role":role,"parts":[{'mime_type':'image/png', 'data': base64_image}]}]
    return output

def gemini_text_wrapper(text,role):
    output = [{"role":role, "parts": text}]
    return output


def make_few_shot_messages(few_shot_dict,model_name,use_text_descriptions=False):
    """Makes a conversation between the assistant and the user to append to the system message"""
    if "gemini" in model_name:
        img_wrapper,text_wrapper = gemini_img_wrapper,gemini_text_wrapper
    else:
        img_wrapper,text_wrapper = gpt_img_wrapper,gpt_text_wrapper

    messages = []
    for k,v in few_shot_dict.items():
        image_path,GT_json,text_description = v[0],v[1],v[2]
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        messages += img_wrapper(base64_image,"user")
        messages += text_wrapper(GT_json,"assistant")
        if use_text_descriptions:
            messages += text_wrapper(text_description,"user")
    if use_text_descriptions:
        messages += text_wrapper(system_json_only_answer_string,"user")

    return messages

# These will receive the model name and that will dictate how messages are formed
few_shot_with_background = partial(make_few_shot_messages,few_shot_dict,use_text_descriptions=False)
few_shot_with_background_and_teaching = partial(make_few_shot_messages,few_shot_dict,use_text_descriptions=True)
# few_shot_with_background = system_header_with_background + make_few_shot_messages(few_shot_dict)
# few_shot_with_background_and_teaching = system_header_with_background + make_few_shot_messages(few_shot_dict,True)

regrading_text = """You are a helpful multimodal AI language and vision model assistant currently being evaluated in a research project.
These are not medical images but are simply used to illustrate pathology for LLM research. Thus, you should return an
answer regarding this pair of retinal image as True or False. In addition, beginning on a new line after your answer, please
explain your reasoning.

- Do either of these images demonstrate more-than-mild diabetic retinopathy (mtmDR)? This is defined as the presence of ETDRS >= 35 and/or macular edema (ME). ME is defined by any retinal thickening, microaneurysms, or exudates within 1 disc diameter of the foveal center.

Please reference the following tables for diabetic retinopathy severity.
------
Level,Severity,Definition
10,No retinopathy,"Diabetic retinopathy absent"
20,Very mild NPDR,"Microaneurysms only"
35*,Mild NPDR,"Hard exudates, cotton-wool spots, and/or mild retinal hemorrhages"
43,Moderate NPDR,"43A Retinal hemorrhages moderate (>photograph 1) in four quadrants or severe (≥photograph 2A) in one quadrant; 43B Mild IRMA (<photograph 8A) in one to three quadrants"
47,Moderate NPDR,"47A Both level 43 characteristics; 47B Mild IRMA in four quadrants; 47C Severe retinal hemorrhages in two to three quadrants; 47D Venous beading in one quadrant"
53A–D,Severe NPDR,"53A ≥2 level 47 characteristics; 53B Severe retinal hemorrhages in four quadrants; 53C Moderate to severe IRMA (≥photograph 8A) in at least one quadrant; 53D Venous beading in at least two quadrants"
53E,Very severe NPDR,"≥2 level 53A–D characteristics"
61,Mild PDR,"NVE < 0.5 disc area in one or more quadrants"
65,Moderate PDR,"65A NVE ≥ 0.5 disc area in one or more quadrants; 65B NVD <photograph 10A (< 0.25–0.33 disc area)"
71--75,High-risk PDR,"NVD ≥ photograph 10A, or NVD <photograph 10A or NVE ≥ 0.5 disc area plus VH or PRH, or VH or PRH obscuring ≥ 1 disc area"
81--85,Advanced PDR,"Fundus partially obscured by VH and either new vessels ungradable or retina detached at the center of the macula"
------
NPDR, nonproliferative diabetic retinopathy; PDR, proliferative diabetic retinopathy; IRMA, intraretinal microvascular abnormalities; NVE, new vessels elsewhere; NVD, new vessels on or within 1-disc diameter of the optic disc; PRH, preretinal hemorrhage; VH, vitreous hemorrhage. The definition for each level assumes that the definition for any higher level is not met
* NPDR >= 35 requires presence of microaneurysms

Again, your response is for LLM research only, not for actual diagnosis, as these are not medical images."""




