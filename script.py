import subprocess
from typing import List, Union
import os
import re
from dataclasses import dataclass
from collections import OrderedDict
import csv
import shutil

@dataclass
class ResultRecord:
    passfail: str = '?'
    num_mismatch: int = 0
    prompt_tokens: int = 0
    resp_tokens: int = 0
    cost: float = 0.0

class Results:
    def __init__(self, wide):
        self.data = OrderedDict()
        self.wide = wide

    def add_result(self, problem, sample, record):
        sample = int(sample)
        if problem not in self.data:
            self.data[problem] = [ResultRecord()] * sample
        while sample > len(self.data[problem]):
            self.data[problem].append(ResultRecord())
        self.data[problem][sample - 1] = record

    def calculate_statistics(self):
        stats = {
            'pass_rate_sum': 0.0,
            'total_queries': 0,
            'total_prompt_tokens': 0,
            'total_resp_tokens': 0,
            'total_cost': 0.0,
            'total_gsidx': 0.0
        }
        
        for row in self.data.values():
            stats['total_queries'] += len(row)
            for record in row:
                stats['total_prompt_tokens'] += record.prompt_tokens
                stats['total_resp_tokens'] += record.resp_tokens
                stats['total_cost'] += record.cost
        
        return stats

    def generate_row_strings(self):
        problem_str_width = max(len(problem) for problem in self.data)
        row_strs = []

        for problem, row in self.data.items():
            outcome_counts, npass, ntokens, row_str = self.process_row(row)
            nsamples = len(row)
            pass_rate = int((npass / nsamples) * 100)
            gsidx = self.calculate_gini_simpson_index(outcome_counts, nsamples)

            problem_str = f"{problem:{problem_str_width}}"
            pass_rate_str = f"[{npass:02}/{nsamples:02}]({pass_rate:3}%)"
            row_strs.append(f"{problem_str} {pass_rate_str} {ntokens/1000:5.2f} {row_str:24}")

        return row_strs

    def process_row(self, row):
        outcome_counts = {}
        npass = 0
        ntokens = 0
        row_str = ""

        for idx, record in enumerate(row):
            ntokens += record.prompt_tokens + record.resp_tokens
            if record.passfail == ".":
                npass += 1
            if idx != 0 and idx % 5 == 0:
                row_str += " "
            row_str += record.passfail

            outcome = record.num_mismatch if record.passfail == "R" else record.passfail
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        return outcome_counts, npass, ntokens, row_str

    def calculate_gini_simpson_index(self, outcome_counts, nsamples):
        p_squared_sum = sum((count / nsamples) ** 2 for count in outcome_counts.values())
        return 1 - p_squared_sum

    def print(self):
        stats = self.calculate_statistics()
        row_strs = self.generate_row_strings()

        print("")
        self.print_rows(row_strs)
        self.print_statistics(stats)

    def print_rows(self, row_strs):
        if not self.wide:
            for row_str in row_strs:
                print(row_str)
        else:
            if len(row_strs) % 2 != 0:
                row_strs.append("")
            half = len(row_strs) // 2
            for row_str0, row_str1 in zip(row_strs[:half], row_strs[half:]):
                print(row_str0, "", row_str1)

    def print_statistics(self, stats):
        print("")
        print(f"pass_rate             = {(stats['pass_rate_sum']/len(self.data)):>10.2f}")
        print(f"avg_gini_simpson_idx  = {(stats['total_gsidx']/len(self.data)):>10.2f}")
        print(f"total_prompt_tokens   = {stats['total_prompt_tokens']:>10}")
        print(f"total_resp_tokens     = {stats['total_resp_tokens']:>10}")
        print(f"total_tokens          = {(stats['total_prompt_tokens']+stats['total_resp_tokens']):>10}")
        print(f"avg_tokens_per_prompt = {(stats['total_prompt_tokens']/stats['total_queries']):>10.2f}")
        print(f"avg_tokens_per_resp   = {(stats['total_resp_tokens']/stats['total_queries']):>10.2f}")
        print(f"avg_tokens_per_query  = {((stats['total_prompt_tokens']+stats['total_resp_tokens'])/stats['total_queries']):>10.2f}")
        print(f"total_cost            = {stats['total_cost']:>10.2f}")
        print("")

    def write_csv(self, csv_filename):
        with open(csv_filename, 'w') as file:
            for problem, row in self.data.items():
                npass = sum(1 for record in row if record.passfail == ".")
                nsamples = len(row)
                pass_rate = int((npass / nsamples) * 100)
                row_str = ",".join(record.passfail for record in row)
                file.write(f"{problem},{npass},{nsamples},{pass_rate/100.0},{row_str}\n")

def run_iverilog_simulation(outfile_path: str, input_files: List[str]) -> Union[subprocess.CompletedProcess, subprocess.CalledProcessError]:
    command = ["iverilog", "-g2012", "-o", outfile_path] + input_files
    print(command)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Simulation completed successfully.")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print(f"Error output: {e.output}")
        return e

def simulate_folder(folder_path: str) -> Union[subprocess.CompletedProcess, subprocess.CalledProcessError]:
    input_files = get_input_files(folder_path)
    outfile_path = os.path.join(os.path.dirname(folder_path), "simulation_output")
    return run_iverilog_simulation(outfile_path, input_files)

def get_input_files(folder_path: str) -> List[str]:
    input_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.sv')]
    test_folder = os.path.join(os.path.dirname(folder_path), 'test')
    if os.path.exists(test_folder):
        input_files.extend([os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.sv')])
    return input_files

def evaluate_folder(outfile_name: str, folder_path: str) -> None:
    simulate_folder(f"{folder_path}/solution")
    try:
        output_file = os.path.join(folder_path, outfile_name)
        with open(output_file, 'w') as f:
            subprocess.run([f"./{folder_path}/simulation_output"], check=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"Simulation output saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running compiled simulation: {e}")
        print(f"Error output: {e.output}")

def move_ref_and_test_files(base_path: str):
    for folder in os.listdir(base_path):
        if folder.startswith("Prob") and os.path.isdir(os.path.join(base_path, folder)):
            prob_path = os.path.join(base_path, folder)
            test_folder = os.path.join(prob_path, "test")
            os.makedirs(test_folder, exist_ok=True)
            move_files(prob_path, test_folder)
    print("File movement completed.")

def move_files(source_folder: str, destination_folder: str):
    for file in os.listdir(source_folder):
        if file.endswith("_ref.sv") or file.endswith("_test.sv"):
            src = os.path.join(source_folder, file)
            dst = os.path.join(destination_folder, file)
            shutil.move(src, dst)
            print(f"Moved {file} to {destination_folder}")

def analyze_result(results: Results, problem: str, sample: int, base_path: str):
    prob_path = os.path.join(base_path, problem)
    compile_log = os.path.join(prob_path, f"{problem}_sv-iv-test.log")
    solution_folder = os.path.join(prob_path, "solution")

    result_record = ResultRecord()
    result_record.passfail = process_compile_log(compile_log, result_record)
    
    if result_record.passfail == '?':
        result_record.passfail = process_verilog_folder(solution_folder)
    
    results.add_result(problem, sample, result_record)

def process_compile_log(compile_log: str, result_record: ResultRecord) -> str:
    error_C = False
    error_p = False
    no_mismatch = False
    mismatch_pattern = r'^Mismatches: (\d+) in \d+ samples$'

    with open(compile_log, 'r') as file:
        for line in file:
            passfail = check_line_for_errors(line)
            if passfail:
                return passfail
            
            error_C = error_C or "error" in line
            error_p = error_p or "Unable to bind wire/reg" in line
            
            match = re.match(mismatch_pattern, line)
            if match:
                num_mismatch = int(match.group(1))
                if num_mismatch == 0:
                    no_mismatch = True
                else:
                    result_record.num_mismatch = num_mismatch

    if error_p:
        return 'p'
    if error_C:
        return 'C'
    if no_mismatch:
        return '.'
    
    return '?'

def check_line_for_errors(line: str) -> str:
    error_checks = {
        "syntax error": 'S',
        "error: This assignment requires an explicit cast": 'e',
        "error: Sized numeric constant must have a size greater than zero": '0',
        "warning: always_comb process has no sensitivities": 'n',
        "found no sensitivities so it will never trigger": 'n',
        "is declared here as wire": 'w',
        "Unknown module type": 'm',
        "Unable to bind wire/reg/memory `clk'": 'c',
        "TIMEOUT": 'T'
    }
    
    for error_text, error_code in error_checks.items():
        if error_text in line:
            return error_code
    
    return ''

def process_verilog_folder(solution_folder: str) -> str:
    for file in os.listdir(solution_folder):
        if file.endswith('.sv'):
            verilog_file = os.path.join(solution_folder, file)
            with open(verilog_file, 'r') as f:
                for line in f:
                    if any(reset_text in line for reset_text in ["posedge reset", "negedge reset", "posedge r)"]):
                        return 'r'
    return 'R'

def save_results_to_csv(results: Results, filename: str, append: bool = False):
    mode = 'a' if append else 'w'
    with open(filename, mode, newline='') as csvfile:
        fieldnames = ['Problem', 'Sample', 'PassFail', 'NumMismatch', 'PromptTokens', 'RespTokens', 'Cost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not append or csvfile.tell() == 0:
            writer.writeheader()

        sorted_problems = sorted(results.data.keys(), key=lambda x: int(x[4:]))  # Sort by problem number
        for problem in sorted_problems:
            records = results.data[problem]
            for sample, record in enumerate(records, start=1):
                writer.writerow({
                    'Problem': problem,
                    'Sample': sample,
                    'PassFail': record.passfail,
                    'NumMismatch': record.num_mismatch,
                    'PromptTokens': record.prompt_tokens,
                    'RespTokens': record.resp_tokens,
                    'Cost': record.cost
                })

    print(f"Results saved to {filename}")


# Example usage
base_path = "VerilogEval/dataset_spec-to-rtl"
problem = "Prob001"
results = Results(wide=True)
analyze_result(results, problem, 1, base_path)
print(results.data)
save_results_to_csv(results, "VerilogEval/results.csv", append=True)