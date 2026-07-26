---
license: mit
task_categories:
- conversational
language:
- en
- zh
- it
- vi
- ar
- ko
- th
- bn
- sw
- jv
size_categories:
- n<1K
---

# Multilingual Jailbreak Challenges in Large Language Models

This repo contains the data for our paper ["Multilingual Jailbreak Challenges in Large Language Models"](https://arxiv.org/abs/2310.06474).
[[Github repo]](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/)

## Annotation Statistics
We collected a total of 315 English unsafe prompts and annotated them into nine non-English languages. The languages were categorized based on resource availability, as shown below:

**High-resource languages:** Chinese (zh), Italian (it), Vietnamese (vi)

**Medium-resource languages:** Arabic (ar), Korean (ko), Thai (th)

**Low-resource languages:** Bengali (bn), Swahili (sw), Javanese (jv)

## Ethics Statement
Our research investigates the safety challenges of LLMs in multilingual settings. We are aware of the potential misuse of our findings and emphasize that our research is solely for academic purposes and ethical use. Misuse or harm resulting from the information in this paper is strongly discouraged. To address the identified risks and vulnerabilities, we commit to open-sourcing the data used in our study. This openness aims to facilitate vulnerability identification, encourage discussions, and foster collaborative efforts to enhance LLM safety in multilingual contexts. Furthermore, we have developed the SELF-DEFENSE framework to address multilingual jailbreak challenges in LLMs. This framework automatically generates multilingual safety training data to mitigate risks associated with unintentional and intentional jailbreak scenarios. Overall, our work not only highlights multilingual jailbreak challenges in LLMs but also paves the way for future research, collaboration, and innovation to enhance their safety.

## Citation
```
@misc{deng2023multilingual,
title={Multilingual Jailbreak Challenges in Large Language Models},
author={Yue Deng and Wenxuan Zhang and Sinno Jialin Pan and Lidong Bing},
year={2023},
eprint={2310.06474},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```