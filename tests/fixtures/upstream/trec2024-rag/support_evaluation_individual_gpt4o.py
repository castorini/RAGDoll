"""
Description: This script is used to evaluate the support for the RAG answers using the OpenAI API.

export AZURE_OPENAI_ENDPOINT="https://trec-rag-2024.openai.azure.com/"
export AZURE_OPENAI_API_KEY="XXXX"

python support_evaluation_gpt4o.py \
       --result_filepath "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/runs/auggen" \
       --output_filepath "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/results/citation_eval/auggen/" \
       --model "gpt-4o" --temperature 0.1 
"""

from openai import AzureOpenAI
import openai
import argparse
import json, os
import time, csv
from tqdm.autonotebook import tqdm
from pyserini.search import LuceneSearcher

SUPPORT_EVAL_PROMPT = """
In this task, you will evaluate whether each statement is supported by its corresponding citations. 
Note that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are not easy to discern at first glance. 
Pay close attention to the text. Read it carefully as you would when proofreading.

You will be provided with a statement and its corresponding citation. It may be helpful to ask yourself whether it is accurate to say "according to the citation" with a
statement following this phrase. Be sure to check all of the information in the statement. You will be given three options:

- "Full Support": All of the information in the statement is supported in the citation.
- "Partial Support": Only some of the information is supported in the citation, but other parts of the information are missing from the citation.
- "No Support": This citation does not support any part of the statement.

Please provide your response based on the information in the citation. If you are unsure, use your best judgment. 
Respond as either "Full Support", "Partial Support", or "No Support" with no additional information.

Statement: {statement}

Citation: {citation}

Response:
"""

def load_qrels(qrels_file):
    query_ids = set()
    reader = csv.reader(open(qrels_file, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    for id, row in enumerate(reader):
        query_id = row[0]
        query_ids.add(query_id)
    
    return list(query_ids)

class AbstractEvaluation:
    def __init__(self):
        self.prompt = None
    
    def __call__(self, *args, **kwargs):
        return self.prompt.format(*args, **kwargs)

class SupportEval(AbstractEvaluation):
    def __init__(self):
        self.prompt = SUPPORT_EVAL_PROMPT
    
    def postprocess(self, response):
        citation = None
        if response:
            if "full support" in response.lower():
                citation = "FS"
            elif "partial support" in response.lower():
                citation = "PS"
            elif "no support" in response.lower():
                citation = "NS"
        return citation

class OpenAIClient:
    def __init__(self, 
                 model: str = None, 
                 endpoint: str = None, 
                 api_key: str = None, 
                 api_version: str = "2024-02-01", 
                 wait: int = 60):
        
        self.deployment_name = "gpt-35-turbo" if "gpt-3.5-turbo" in model else model
        self.wait = wait
        self.client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if endpoint is None else endpoint, 
            api_key=os.getenv("AZURE_OPENAI_API_KEY") if api_key is None else api_key,  
            api_version=api_version,
        )

    def chat(self, prompt: str, temperature: float, n: int):

        try:
            response = self.client.chat.completions.create(
                    model=self.deployment_name, # model = "deployment_name".
                    messages=[{"role": "user", "content": f"{prompt}"}],
                    temperature=temperature,
                    n=n,
            )
            output = response.choices[0].message.content
            return output
        
        except openai.RateLimitError as e:
            retry_time = int(str(e).split("retry after")[1].split('second')[0].strip())            
            time.sleep(retry_time)
            return self.response(prompt, temperature, n)
        
        except openai.InternalServerError as e:
            return None
        
        except IndexError as e:
            return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--result_filepath", required=True, type=str, help="Path to the file containing the RAG answers")
    parser.add_argument("--output_filepath", required=True, type=str, help="Path to the output file")
    parser.add_argument("--qrels_filepath", required=True, type=str, help="Path to the qrels file")
    parser.add_argument("--model", required=True, type=str, default="gpt-4o", choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4"])
    parser.add_argument("--top_k", required=False, type=int, default=20)
    parser.add_argument("--temperature", required=False, type=float, default=0.1)
    parser.add_argument("--num_sequence", required=False, type=int, default=1)
    parser.add_argument("--lucene_index", required=False, type=str, default="/store/scratch/rpradeep/nuggetizer/data/indexes/lucene-inverted.msmarco-v2.1-doc-segmented")
    parser.add_argument("--max_citations", required=False, type=int, default=3)

    args = parser.parse_args()

    # FILTER_SET = [
    #     "InfoLab.UDInfolab.AG-v1",
    #     "InfoLab.UDInfolab.AG-v2",
    #     "KML.gpt_mini_double_prompt.jsonl"
    # ]
    
    # Initialize the OpenAI client
    print(f"Loading OpenAI model: {args.model}")
    client = OpenAIClient(model=args.model)

    # Initialize the Prompt for Citation Evaluation
    print("Initializing Support Evaluation Prompt...")
    cite_eval_prompt = SupportEval()

    # walk through the results file
    input_filepaths = os.listdir(args.result_filepath)
    os.makedirs(args.output_filepath, exist_ok=True)

    # pyserini load index
    seg_index = LuceneSearcher(args.lucene_index)

    # Load qrels filepath
    topics = load_qrels(args.qrels_filepath)

    # Round 1: evaluate whether sentence requires a citation or not
    for input_filepath in tqdm(input_filepaths, total=len(input_filepaths), desc="All Submissions: "):

        topics_done = set()

        # check if the file is already generated or not (to resume the evaluation)
        output_filename = os.path.join(args.output_filepath, f"{input_filepath}.jsonl")

        if os.path.exists(output_filename):
            with open(output_filename, "r") as f:
                for line in f:
                    row = json.loads(line)
                    topics_done.add(row['topic_id'])
            
            if len(topics_done) < len(topics):
                print(f"Resuming {len(topics) - len(topics_done)} queries for {input_filepath} ....")
            else:
                print(f"Skipping {input_filepath} as it is already processed with all topics!")
                continue
        
        # continue with the evaluation
        missed_topics = list(set(topics) - topics_done)
        print(f"Processing {len(missed_topics)} topics.....")
        
        num_lines = sum(1 for _ in open(os.path.join(args.result_filepath, input_filepath), 'rb'))
        with open(os.path.join(args.result_filepath, input_filepath), 'r') as fin:
            with open(output_filename, 'a') as fout:
                for line in tqdm(fin, total=num_lines, desc="All Topics:"):                    
                    corpus = {}
                    data = json.loads(line)
                    
                    # either the topic_id is not in the missed_topics or the answer is empty
                    if data['topic_id'] not in missed_topics or len(data["answer"]) == 0:
                        continue
                    
                    else:
                        document_ids = data['references']
                        for doc_id in document_ids:
                            if doc_id not in corpus:
                                seg = json.loads(seg_index.doc(doc_id).raw())
                                corpus[doc_id] = seg['title'] + ': ' + seg['segment']
                        
                        data['segments'] = corpus
                        data["support_eval"] = []
                        
                        for answer in data["answer"]:
                            if answer["citations"] is []:
                                continue
                            responses, support_scores = [], []
                            citations = [document_ids[citation] for citation in answer["citations"][:args.max_citations]]
                            
                            for document_id in citations:
                                prompt = cite_eval_prompt(statement=answer["text"], citation=corpus[document_id])
                                response = client.chat(
                                    prompt=prompt, 
                                    temperature=args.temperature, 
                                    n=args.num_sequence
                                )
                                support_score = cite_eval_prompt.postprocess(response)
                                support_scores.append(support_score)
                                responses.append(response)
                            
                            data["support_eval"].append({
                                "answer": answer["text"],
                                "citations": answer["citations"],
                                "count": len(answer["citations"]),
                                "eval_scores": support_scores,
                                f"{args.model}_responses": responses
                            })
                        
                        # write the data to the output file
                        data['segments'] = corpus
                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                        fout.flush()