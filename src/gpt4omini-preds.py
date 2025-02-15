from dotenv import load_dotenv
import pandas as pd
import openai
import ast
import json
import os
import re

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
client = openai.OpenAI()

input_path = ""
output_path = ""
output_final_csv_path = ""

### DOMAINS: 
# 0-> URW
# 1-> CC
# 2-> Other

# Uncomment and adjust the following block if need to load and concatenate multiple fold files
# dataframes = []
# for i in range(1, 6):  # from fold1 to fold5
#     file_path = os.path.join(input_path, f'fold{i}_predictions.csv')
#     aux = pd.read_csv(file_path)
#     dataframes.append(aux)
# df = pd.concat(dataframes, ignore_index=True)
# print(len(df))
# df.head()

file_path = os.path.join("")
df = pd.read_csv(file_path)
df = df[df["language"] == "RU"]

print(len(df))
df.head()

def get_climate_prompt(text):
    return f'''In the following text, identify the core narrative that aligns with the author's perspective. 
Classify it based on the options in the list below. If the narrative in the text fall outside the list, answer "Other".

### Options List (Narratives and subnarratives)
Criticism of climate policies
- Climate policies are ineffective
- Climate policies have negative impact on the economy
- Climate policies are only for profit
- Other

Criticism of institutions and authorities
- Criticism of the EU
- Criticism of international entities
- Criticism of national governments
- Criticism of political organizations and figures
- Other

Climate change is beneficial
- CO2 is beneficial
- Temperature increase is beneficial
- Other

Downplaying climate change
- Climate cycles are natural
- Weather suggests the trend is global cooling
- Temperature increase does not have significant impact
- CO2 concentrations are too small to have an impact
- Human activities do not impact climate change
- Ice is not melting
- Sea levels are not rising
- Humans and nature will adapt to the changes
- Other

Questioning the measurements and science
- Methodologies/metrics used are unreliable/faulty
- Data shows no temperature increase
- Greenhouse effect/carbon dioxide do not drive climate change
- Scientific community is unreliable
- Other

Criticism of climate movement
- Climate movement is alarmist
- Climate movement is corrupt
- Ad hominem attacks on key activists
- Other

Controversy about green technologies
- Renewable energy is dangerous
- Renewable energy is unreliable
- Renewable energy is costly
- Nuclear energy is not climate friendly
- Other

Hidden plots by secret schemes of powerful groups
- Blaming global elites
- Climate agenda has hidden motives
- Other

Amplifying Climate Fears
- Earth will be uninhabitable soon
- Amplifying existing fears of global warming
- Doomsday scenarios for humans
- Whatever we do it is already too late
- Other

Green policies are geopolitical instruments
- Climate-related international relations are abusive/exploitative
- Green activities are a form of neo-colonialism
- Other

### Text
             
{text}

Provide only the most relevant narratives that best fit the text's intent. 
If multiple narratives are equally significant, include them all.
Answer **only** with the classifications and always include narratives and subnarratives.'''

def get_war_prompt(text):
    return f'''In the following text, identify the core narrative that aligns with the author's perspective. 
Classify it based on the options in the list below. If the narrative in the text fall outside the list, answer "Other".

### Options List (Narratives and subnarratives)
Blaming the war on others
- Ukraine is the aggressor
- The West are the aggressors
- Other

Discrediting Ukraine
- Rewriting Ukraine’s history
- Discrediting Ukrainian nation and society
- Discrediting Ukrainian military
- Discrediting Ukrainian government and officials and policies
- Ukraine is a puppet of the West
- Ukraine is a hub for criminal activities
- Ukraine is associated with nazism
- Situation in Ukraine is hopeless
- Other

Russia is the Victim
- The West is russophobic
- Russia actions in Ukraine are only self-defence
- UA is anti-RU extremists
- Other

Praise of Russia
- Praise of Russian military might
- Praise of Russian President Vladimir Putin
- Russia is a guarantor of peace and prosperity
- Russia has international support from a number of countries and people
- Russian invasion has strong national support
- Other

Overpraising the West
- NATO will destroy Russia
- The West belongs in the right side of history
- The West has the strongest international support
- Other

Speculating war outcomes
- Russian army is collapsing
- Russian army will lose all the occupied territories
- Ukrainian army is collapsing
- Other

Discrediting the West, Diplomacy
- The EU is divided
- The West is weak
- The West is overreacting
- The West does not care about Ukraine, only about its interests
- Diplomacy does/will not work
- West is tired of Ukraine
- Other

Negative Consequences for the West
- Sanctions imposed by Western countries will backfire
- The conflict will increase the Ukrainian refugee flows to Europe
- Other

Distrust towards Media
- Western media is an instrument of propaganda
- Ukrainian media cannot be trusted
- Other

Amplifying war-related fears
- By continuing the war we risk WWIII
- Russia will also attack other countries
- There is a real possibility that nuclear weapons will be employed
- NATO should/will directly intervene
- Other

### Text
             
{text}

Provide only the **most** relevant narratives that best fit the text's intent. 
If multiple narratives are equally significant, include them all.
Answer **only** with the classifications and always include narratives and subnarratives.'''

def get_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.95,
        max_completion_tokens=200
    )
    return completion.choices[0].message

generations = []

for idx, row in df.iterrows():
    article_id = row['article_id']
    content = row['content']
    language = row['language']
    p_domains = row['p_domains']
    
    if row['p_domains'] == 0:
        response = get_response(get_war_prompt(content))
    elif row['p_domains'] == 1:
        response = get_response(get_climate_prompt(content))
    else:
        response = 'OTHEROTHER'
    
    generated_texts = response if isinstance(response, str) else response.content
    
    result = {
        "article_id": article_id,
        "content": content,
        "language": language,
        "p_domains": p_domains,
        "gpt_output": generated_texts
    }
    
    generations.append(result)
    
    os.makedirs(output_path, exist_ok=True)
    output_name = 'gpt-4o-mini'
    if '/' in output_name:
        output_name = output_name.split('/')[-1]
        
    json_output_path = os.path.join(output_path, f"{output_name}_intermediate.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(generations, f, ensure_ascii=False, indent=4)

results = pd.DataFrame(generations)

narratives = {
    "Criticism of climate policies": [
        "Climate policies are ineffective",
        "Climate policies have negative impact on the economy",
        "Climate policies are only for profit",
        "Other"
    ],
    "Criticism of institutions and authorities": [
        "Criticism of the EU",
        "Criticism of international entities",
        "Criticism of national governments",
        "Criticism of political organizations and figures",
        "Other"
    ],
    "Climate change is beneficial": [
        "CO2 is beneficial",
        "Temperature increase is beneficial",
        "Other"
    ],
    "Downplaying climate change": [
        "Climate cycles are natural",
        "Weather suggests the trend is global cooling",
        "Temperature increase does not have significant impact",
        "CO2 concentrations are too small to have an impact",
        "Human activities do not impact climate change",
        "Ice is not melting",
        "Sea levels are not rising",
        "Humans and nature will adapt to the changes",
        "Other"
    ],
    "Questioning the measurements and science": [
        "Methodologies/metrics used are unreliable/faulty",
        "Data shows no temperature increase",
        "Greenhouse effect/carbon dioxide do not drive climate change",
        "Scientific community is unreliable",
        "Other"
    ],
    "Criticism of climate movement": [
        "Climate movement is alarmist",
        "Climate movement is corrupt",
        "Ad hominem attacks on key activists",
        "Other"
    ],
    "Controversy about green technologies": [
        "Renewable energy is dangerous",
        "Renewable energy is unreliable",
        "Renewable energy is costly",
        "Nuclear energy is not climate friendly",
        "Other"
    ],
    "Hidden plots by secret schemes of powerful groups": [
        "Blaming global elites",
        "Climate agenda has hidden motives",
        "Other"
    ],
    "Amplifying Climate Fears": [
        "Earth will be uninhabitable soon",
        "Amplifying existing fears of global warming",
        "Doomsday scenarios for humans",
        "Whatever we do it is already too late",
        "Other"
    ],
    "Green policies are geopolitical instruments": [
        "Climate-related international relations are abusive/exploitative",
        "Green activities are a form of neo-colonialism",
        "Other"
    ],
    "Blaming the war on others": [
        "Ukraine is the aggressor",
        "The West are the aggressors",
        "Other"
    ],
    "Discrediting Ukraine": [
        "Rewriting Ukraine’s history",
        "Discrediting Ukrainian nation and society",
        "Discrediting Ukrainian military",
        "Discrediting Ukrainian government and officials and policies",
        "Ukraine is a puppet of the West",
        "Ukraine is a hub for criminal activities",
        "Ukraine is associated with nazism",
        "Situation in Ukraine is hopeless",
        "Other"
    ],
    "Russia is the Victim": [
        "The West is russophobic",
        "Russia actions in Ukraine are only self-defence",
        "UA is anti-RU extremists",
        "Other"
    ],
    "Praise of Russia": [
        "Praise of Russian military might",
        "Praise of Russian President Vladimir Putin",
        "Russia is a guarantor of peace and prosperity",
        "Russia has international support from a number of countries and people",
        "Russian invasion has strong national support",
        "Other"
    ],
    "Overpraising the West": [
        "NATO will destroy Russia",
        "The West belongs in the right side of history",
        "The West has the strongest international support",
        "Other"
    ],
    "Speculating war outcomes": [
        "Russian army is collapsing",
        "Russian army will lose all the occupied territories",
        "Ukrainian army is collapsing",
        "Other"
    ],
    "Discrediting the West, Diplomacy": [
        "The EU is divided",
        "The West is weak",
        "The West is overreacting",
        "The West does not care about Ukraine, only about its interests",
        "Diplomacy does/will not work",
        "West is tired of Ukraine",
        "Other"
    ],
    "Negative Consequences for the West": [
        "Sanctions imposed by Western countries will backfire",
        "The conflict will increase the Ukrainian refugee flows to Europe",
        "Other"
    ],
    "Distrust towards Media": [
        "Western media is an instrument of propaganda",
        "Ukrainian media cannot be trusted",
        "Other"
    ],
    "Amplifying war-related fears": [
        "By continuing the war we risk WWIII",
        "Russia will also attack other countries",
        "There is a real possibility that nuclear weapons will be employed",
        "NATO should/will directly intervene",
        "Other"
    ]
}

def detect_narratives(text):
    detected_narrative = set()
    detected_subnarrative = set()
    
    for narrative, subnarratives in narratives.items():
        narrative_detected = False
        subnarrative_detected = False
        
        if re.search(r'\b' + re.escape(narrative) + r'\b', text, re.IGNORECASE):
            narrative_detected = True
            detected_narrative.add(narrative)
        
            if subnarratives:
                for subnarrative in subnarratives:
                    if re.search(r'\b' + re.escape(subnarrative) + r'\b', text, re.IGNORECASE):
                        detected_subnarrative.add(subnarrative)
                        subnarrative_detected = True
                
                if not subnarrative_detected:
                    if 'Other' not in subnarratives:
                        adjusted_subnarrative = f"{narrative}: Other"
                        detected_subnarrative.add(adjusted_subnarrative)
            else:
                adjusted_subnarrative = f"{narrative}: Other"
                detected_subnarrative.add(adjusted_subnarrative)
    
    if not detected_narrative:
        detected_narrative.add("Other")
    if not detected_subnarrative:
        detected_subnarrative.add("Other")
        
    return {
        "narrative": list(detected_narrative),
        "subnarrative": list(detected_subnarrative)
    }

p_system = [
    detect_narratives(row['gpt_output']) if row['p_domains'] != 2 else {'narrative': ['Other'], 'subnarrative': ['Other']}
    for _, row in results.iterrows()
]

print(sum(1 for item in p_system if item == {'narrative': ['Other'], 'subnarrative': ['Other']}))
print(sum(1 for item in p_system if item != {'narrative': ['Other'], 'subnarrative': ['Other']}))
print(sum(1 for item in p_system if item == {'narrative': ['Other'], 'subnarrative': ['Other']}) + sum(1 for item in p_system if item != {'narrative': ['Other'], 'subnarrative': ['Other']}))

results['p_system'] = p_system

results.to_csv(output_final_csv_path, index=False, encoding='utf-8')

print(f"CSV file created at: {output_final_csv_path}")

true_labels = results["labels"].apply(ast.literal_eval).tolist()
true_narratives = [set(item['narrative']) for item in true_labels]
true_subnarratives = [set(item['subnarrative']) for item in true_labels]

pred_narratives = [set(item['narrative']) for item in results["p_system"]]
pred_subnarratives = [set(item['subnarrative']) for item in results["p_system"]]
