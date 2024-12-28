#CA
import matplotlib
matplotlib.use('Agg')
import heapq
import json
import os
import gc
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import threading

def network_creation():
    G = nx.Graph()
    nodes = range(1, 25)
    G.add_nodes_from(nodes)
    edges = [
        (1, 2, float(100)), (1, 6, float(110)), 
        (2, 3, float(90)), (2, 6, float(95)),
        (3, 4, float(85)), (3, 7, float(100)), (3, 5, float(95)),
        (4, 5, float(90)), (4, 7, float(70)), 
        (5, 8, float(120)), 
        (6, 7, float(100)), (6, 9, float(110)), (6, 11, float(130)),
        (7, 8, float(95)), (7, 9, float(90)), 
        (8, 10, float(85)), 
        (9, 12, float(100)), (9, 11, float(120)), (9, 12, float(110)),
        (10, 14, float(80)), (10, 13, float(90)),
        (11, 12, float(85)), (11, 15, float(120)), (11, 19, float(150)),
        (12, 13, float(75)), (12, 16, float(95)),
        (13, 14, float(70)), (13, 17, float(85)),
        (14, 18, float(100)),
        (15, 16, float(80)), (15, 20, float(110)),
        (16, 17, float(85)), (16, 22, float(90)), (16, 21, float(95)),
        (17, 18, float(80)), (17, 22, float(85)), (17, 23, float(95)),
        (18, 24, float(90)),
        (19, 20, float(100)),
        (20, 21, float(75)),
        (21, 22, float(60)), 
        (22, 23, float(70)),
        (23, 24, float(85))
    ]
    G.add_weighted_edges_from(edges, weight='BW')
    for node in G.nodes:
        G.nodes[node]['CPU'] = float(random.uniform(2000, 3000))
    return G

def check_resources(network, path, functions_detail, bw, delay_threshold, queueing_delay, relaxation_factor=1.5):
    adjusted_delay_threshold = delay_threshold * relaxation_factor - queueing_delay
    calculated_delay = calculate_delays(functions_detail, len(path))
    if calculated_delay > adjusted_delay_threshold:
        logging.info(f"Path delay: {calculated_delay}, Threshold: {adjusted_delay_threshold}")
        return False
    
    for func in functions_detail:
        required_cpu = func['Required CPU']
        for node in path:
            if network.nodes[node]['CPU'] < required_cpu * relaxation_factor:
                logging.info(f"Node {node} CPU tight: Required {required_cpu}, Available {network.nodes[node]['CPU']}")
                return False
    
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        if network.edges[edge]['BW'] < bw * relaxation_factor:
            logging.info(f"Bandwidth tight on edge {edge}: Required {bw}, Available {network.edges[edge]['BW']}")
            return False
    
    return True

def best_path(network, start, end, functions_detail, bw, delay_threshold, queueing_delay, max_paths=10):
    try:
        all_paths = list(nx.all_simple_paths(network, source=start, target=end, cutoff=None))[:max_paths]
        path_strategies = [
            lambda path: check_resources(
                network, path, functions_detail, bw, 
                delay_threshold, queueing_delay, 
                relaxation_factor=1.0
            ),
            lambda path: check_resources(
                network, path, functions_detail, bw, 
                delay_threshold * 1.2, queueing_delay, 
                relaxation_factor=1.2
            )
        ]
        def path_evaluation(path):
            scores = []
            for strategy in path_strategies:
                if strategy(path):
                    delay = calculate_delays(functions_detail, len(path))
                    length = len(path)
                    score = (
                        delay * 0.6 +  
                        length * 0.4   
                    )
                    scores.append(score)
            return min(scores) if scores else float('inf')
        feasible_paths = sorted(all_paths, key=path_evaluation)
        return feasible_paths[0] if feasible_paths else None
    
    except Exception as e:
        logging.error(f"Error in advanced_path_explorer: {str(e)}")
        return None

def update_resources(network, path, functions_detail, bw, recovery_rate=0.1):
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        current_bw = network.edges[edge]['BW']
        recovery_amount = bw * recovery_rate
        network.edges[edge]['BW'] = float(min(current_bw + recovery_amount, 150)) 
    
    for node in path:
        required_cpu = float(sum(func['Required CPU'] for func in functions_detail))
        current_cpu = network.nodes[node]['CPU']
        recovery_amount = required_cpu * recovery_rate
        network.nodes[node]['CPU'] = float(min(current_cpu + recovery_amount, 3000))  
def ecg_patient_tasks():
    return {
        'ECG_Scenarios': {
            'Routine_Monitoring': [
                'ECG_F1', 'ECG_F2', 'ECG_F3', 'ECG_F4', 'ECG_F5'
            ],
            'Emergency': [
                'ECG_F1', 'ECG_F2', 'ECG_F5'
            ],
            'Archival_Research': [
                'ECG_F1', 'ECG_F4', 'ECG_F5'
            ],
            'Comprehensive': [
                'ECG_F1', 'ECG_F2', 'ECG_F3', 'ECG_F4', 'ECG_F5'
            ]
        },
        'Function_Details': {
            'ECG_F1': ('ECG Monitoring', (40, 120), 800),
            'ECG_F2': ('QRS Detection', (20, 100), 1000),
            'ECG_F3': ('Arrhythmia Detection', (50, 200), 1500),
            'ECG_F4': ('ECG Compression', (10, 50), 1000),
            'ECG_F5': ('ECG Data Transmission', (300, 1500), 500)
        }
    }

def function_cpu_requirements():
    tasks = ecg_patient_tasks()['Function_Details']
    return {fn: details[2] for fn, details in tasks.items()}

def generate_data_points(num_points, filename, criticalness_model):
    data_points = []
    tasks = ecg_patient_tasks()
    cpu_reqs = function_cpu_requirements()
    scenarios = tasks['ECG_Scenarios']

    for i in range(num_points):
        scenario_name = random.choice(list(scenarios.keys()))
        use_functions = scenarios[scenario_name]
        
        start = random.randint(1, 24)
        end = random.randint(1, 24)
        while start == end:
            end = random.randint(1, 24)
            
        features = []
        functions_detail = []
        
        for fn in use_functions:
            task_details = tasks['Function_Details'][fn]
            task_name, value_range, _ = task_details
            measured_value = random.uniform(*value_range)
            features.append((task_name, measured_value))
        
        criticalness_scores = criticalness_model.predict(features)
        
        for idx, fn in enumerate(use_functions):
            task_details = tasks['Function_Details'][fn]
            task_name, _, cpu_req = task_details
            functions_detail.append({
                'Function Name': task_name,
                'Function Code': fn,
                'Scenario': scenario_name,
                'Required CPU': cpu_req,
                'Measured Value': features[idx][1],
                'Criticalness': criticalness_scores[idx]
            })
        
        data_point = {
            'Data Point Number': i + 1,
            'Start': start,
            'End': end,
            'Scenario': scenario_name,
            'Functions Detail': functions_detail,
            'ECG Data Details': {
                'Cleaned ECG Length': random.randint(800, 1200),
                'QRS Detected Length': random.randint(600, 1000),
                'Compressed ECG Length': random.randint(400, 800),
                'Anomalies Detected': random.randint(0, 10),
                'Transmission Delay': random.uniform(100, 500)
            },
            'Bandwidth': random.randint(10, 20),
            'Delay Threshold': random.randint(50000, 65000),
            'Timestamp': str(int((datetime.now() + timedelta(milliseconds=i)).timestamp() * 1000))
        }
        data_points.append(data_point)
    
    with open(filename, 'w') as file:
        json.dump(data_points, file, default=str)
    
    return data_points

def load_data_points(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def extract_features_and_targets(data_points):
    features = []
    targets = []  
    for data in data_points:
        feature_dict = {
            'bw': data['Bandwidth'],
            'delay_threshold': data['Delay Threshold'],
            'function_count': len(data['Functions Detail']),
            'total_required_cpu': sum(f['Required CPU'] for f in data['Functions Detail']),
            'total_criticalness': sum(f['Criticalness'] for f in data['Functions Detail']),
            'max_criticalness': max(f['Criticalness'] for f in data['Functions Detail']),
            'min_criticalness': min(f['Criticalness'] for f in data['Functions Detail']),
            'average_criticalness': sum(f['Criticalness'] for f in data['Functions Detail']) / len(data['Functions Detail']),
            'cleaned_ecg_length': data['ECG Data Details']['Cleaned ECG Length'],
            'qrs_detected_length': data['ECG Data Details']['QRS Detected Length'],
            'compressed_ecg_length': data['ECG Data Details']['Compressed ECG Length'],
            'anomalies_detected': data['ECG Data Details']['Anomalies Detected'],
            'transmission_delay': data['ECG Data Details']['Transmission Delay']
        }
        features.append(list(feature_dict.values()))
        
        avg_criticalness = feature_dict['average_criticalness']
        anomalies = feature_dict['anomalies_detected']
        transmission_delay = feature_dict['transmission_delay']
        priority_score = (
            0.4 * avg_criticalness + 
            0.3 * (anomalies / 10) +
            0.3 * (1 - min(transmission_delay / 500, 1))
        )
        
        #some randomness 
        priority_score += random.uniform(-0.1, 0.1)
        targets.append(1 if priority_score > 0.6 else 0)
    
    columns = list(feature_dict.keys())
    return pd.DataFrame(features, columns=columns), targets

def calculate_delays(functions_detail, path_length):
    processing_delay = float(sum(func['Required CPU'] for func in functions_detail))
    link_delay = float(path_length)
    forwarding_delay = float(path_length - 1) * 0.05
    total_delay = processing_delay + link_delay + forwarding_delay
    return float(total_delay)


class TaskQueue:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.stats = {
            'high_priority_added': 0,
            'low_priority_added': 0,
            'high_priority_processed': 0,
            'low_priority_processed': 0
        }

    def calculate_priority_score(self, remaining_time, is_high_priority, waiting_time):
      
        time_urgency = 1.0 / max(remaining_time, 1)  
        waiting_factor = waiting_time / 1000  
        priority_factor = 1.2 if is_high_priority else 1.0
        waiting_weight = 2.0
        return (priority_factor * time_urgency) + (waiting_weight * waiting_factor)

    def add_to_queue(self, data_point, queue_decision_model):
        try:
            functions_detail = data_point['Functions Detail']
            criticalness_values = [float(func['Criticalness']) for func in functions_detail]
            
            feature_dict = {
                'bw': float(data_point['Bandwidth']),
                'delay_threshold': float(data_point['Delay Threshold']),
                'function_count': float(len(functions_detail)),
                'total_required_cpu': float(sum(func['Required CPU'] for func in functions_detail)),
                'total_criticalness': float(sum(criticalness_values)),
                'max_criticalness': float(max(criticalness_values)),
                'min_criticalness': float(min(criticalness_values)),
                'average_criticalness': float(sum(criticalness_values) / len(criticalness_values)),
                'cleaned_ecg_length': float(data_point['ECG Data Details']['Cleaned ECG Length']),
                'qrs_detected_length': float(data_point['ECG Data Details']['QRS Detected Length']),
                'compressed_ecg_length': float(data_point['ECG Data Details']['Compressed ECG Length']),
                'anomalies_detected': float(data_point['ECG Data Details']['Anomalies Detected']),
                'transmission_delay': float(data_point['ECG Data Details']['Transmission Delay'])
            }
            
            features = pd.DataFrame([feature_dict])
            prediction = queue_decision_model.predict(features)[0]
            is_high_priority = prediction == 1
            current_time = float(time.time() * 1000)  # Convert to milliseconds
            timestamp = float(data_point['Timestamp'])  # Ensure timestamp is float
            remaining_time = float(data_point['Delay Threshold']) - (current_time - timestamp)
            entry_time = datetime.now()
            
            request_info = {
                'remaining_time': remaining_time,
                'timestamp': timestamp,
                'data_point': data_point,
                'entry_time': entry_time,
                'is_high_priority': is_high_priority
            }
            
            with self.lock:
                self._insert_sorted(self.queue, request_info)
                if is_high_priority:
                    self.stats['high_priority_added'] += 1
                else:
                    self.stats['low_priority_added'] += 1
                    
        except Exception as e:
            logging.error(f"Error in add_to_queue: {str(e)}")
            logging.error(f"Data point timestamp: {data_point['Timestamp']}, type: {type(data_point['Timestamp'])}")

    def _insert_sorted(self, queue, request_info):

        waiting_time = (datetime.now() - request_info['entry_time']).total_seconds() * 1000
        new_score = self.calculate_priority_score(
            request_info['remaining_time'],
            request_info['is_high_priority'],
            waiting_time
        )
        

        index = len(queue)
        for i, existing_item in enumerate(queue):
            existing_waiting_time = (datetime.now() - existing_item['entry_time']).total_seconds() * 1000
            existing_score = self.calculate_priority_score(
                existing_item['remaining_time'],
                existing_item['is_high_priority'],
                existing_waiting_time
            )
            if new_score > existing_score:
                index = i
                break
        queue.insert(index, request_info)

    def solve_requests(self, network):
        solved_requests = {'high': 0, 'low': 0}
        processing_times = []
        paths_taken = []

        consecutive_high_priority = 0
        max_consecutive_high_priority = 3 
        
        while True:
            with self.lock:
                if not self.queue:
                    break
                
              
                request_index = 0
                request_info = None
                
                
                if consecutive_high_priority >= max_consecutive_high_priority:
                    for i, req in enumerate(self.queue):
                        if not req['is_high_priority']:
                            request_index = i
                            request_info = req
                            break
                
                if request_info is None and self.queue:
                    request_info = self.queue[0]
                    request_index = 0
                
                if request_info is None:
                    break
                    
     
                self.queue.pop(request_index)
                data_point = request_info['data_point']
                entry_time = request_info['entry_time']
                is_high_priority = request_info['is_high_priority']
            
            try:
                queueing_delay = (datetime.now() - entry_time).total_seconds()
                path = best_path(
                    network,
                    data_point['Start'],
                    data_point['End'],
                    data_point['Functions Detail'],
                    data_point['Bandwidth'],
                    data_point['Delay Threshold'],
                    queueing_delay
                )
                
                if path:
                    update_resources(
                        network,
                        path,
                        data_point['Functions Detail'],
                        data_point['Bandwidth']
                    )
                    
     
                    if is_high_priority:
                        solved_requests['high'] += 1
                        self.stats['high_priority_processed'] += 1
                        consecutive_high_priority += 1
                    else:
                        solved_requests['low'] += 1
                        self.stats['low_priority_processed'] += 1
                        consecutive_high_priority = 0  
                    
                    processing_times.append(datetime.now() - entry_time)
                    paths_taken.append({
                        'request_id': str(data_point['Timestamp']),
                        'path': path,
                        'priority': 'high' if is_high_priority else 'low'
                    })
                    
            except Exception as e:
                logging.error(f"Error processing request: {str(e)}")
                continue
        
        average_time = sum((t.total_seconds() for t in processing_times), 0.0) / len(processing_times) if processing_times else 0
        return solved_requests, average_time, paths_taken

    def get_queue_sizes(self):
        """Return current size of the queue"""
        with self.lock:
            high_priority = sum(1 for item in self.queue if item['is_high_priority'])
            return {
                'high_priority': high_priority,
                'low_priority': len(self.queue) - high_priority
            }

# 1-3 low, medium, high criticality - resource in paper
class CriticalnessModel:
    def predict(self, X):
        criticality = []
        for data in X:
            task, value = data
            if task == 'ECG Monitoring':
                # Simulated heart rate value (from ECG signal analysis)
                if 60 <= value <= 100:
                    criticality.append(1)  # Low - Normal range
                elif 40 <= value < 60 or 101 <= value <= 120:
                    criticality.append(2)  # Medium - Slightly abnormal ranges
                else:
                    criticality.append(3)  # High - Critical ranges
            
            elif task == 'QRS Detection':
                # Value could represent the number of detected QRS complexes per minute
                if 60 <= value <= 100:
                    criticality.append(1)  # Low - Normal QRS detection rate
                elif 40 <= value < 60 or 101 <= value <= 120:
                    criticality.append(2)  # Medium - Potential irregularities
                else:
                    criticality.append(3)  # High - Critical QRS detection (potential arrhythmia)
            
            elif task == 'Arrhythmia Detection':
                # Value could be a percentage representing the presence of irregular beats in the ECG data
                if 0 <= value <= 5:
                    criticality.append(1)  # Low - Minimal irregularities
                elif 6 <= value <= 15:
                    criticality.append(2)  # Medium - Noticeable irregularities
                else:
                    criticality.append(3)  # High - Significant arrhythmia detected
            
            elif task == 'ECG Compression':
                # Value could represent the compression ratio (higher is more compressed)
                if 2 <= value <= 5:
                    criticality.append(1)  # Low - Acceptable compression ratios
                elif 1 <= value < 2 or 5 < value <= 7:
                    criticality.append(2)  # Medium - Potential data loss due to compression
                else:
                    criticality.append(3)  # High - High data loss due to over-compression
            
            elif task == 'ECG Data Transmission':
                # Value represents transmission delay in milliseconds
                if 0 <= value <= 1000:
                    criticality.append(1)  # Low - Fast transmission
                elif 1001 <= value <= 2000:
                    criticality.append(2)  # Medium - Acceptable delay for non-critical data
                else:
                    criticality.append(3)  # High - Delay too long, could risk timely intervention
            
            else:
                criticality.append(2)

        return criticality

def generate_and_prepare_training_data(criticalness_model, training_data_file):
    generate_data_points(1000, training_data_file, criticalness_model)
    training_data_points = load_data_points(training_data_file)
    return extract_features_and_targets(training_data_points)

def train_models(features, targets):
    #split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, 
            min_samples_split=5, min_samples_leaf=2, 
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=10,
            learning_rate=0.1, random_state=42
        ),
       'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=10,
        learning_rate=0.1, random_state=42,
        min_samples_split=5, min_samples_leaf=2
    )
    }
    
    results = {}
    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'feature_importance': pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
    
    return results, X_test, y_test

def track_model_training(model, X_train, y_train, X_val, y_val):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    n_trees = model.get_params()['n_estimators']
    for i in range(1, n_trees + 1):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)

        train_pred_proba = model.predict_proba(X_train)
        val_pred_proba = model.predict_proba(X_val)
        train_losses.append(log_loss(y_train, train_pred_proba))
        val_losses.append(log_loss(y_val, val_pred_proba))
        
        train_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
        val_accuracies.append(accuracy_score(y_val, model.predict(X_val)))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_queue_decision_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    logging.info(f"Queue Decision Model Accuracy: {accuracy}")
    logging.info(f"Queue Decision Classification Report: \n{class_report}")
    return accuracy, class_report

def setup_logging():
    logging.basicConfig(filename='model_performance.log', level=logging.INFO)

def log_results(log_filename, total_requests, solved_requests, average_time, paths_taken, network, model_results):
    try:
        with open(log_filename, 'a') as log_file:

            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"Enhanced Processing Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*50}\n\n")

            log_file.write("Model Comparison Results:\n")
            log_file.write(f"{'Model':<15} {'Accuracy':<10} {'AUC-ROC':<10} {'Avg Precision':<15}\n")
            log_file.write("-" * 50 + "\n")
            for name, results in model_results.items():
                log_file.write(f"{name:<15} {results['accuracy']:.4f} {results['auc_roc']:.4f} {results['avg_precision']:.4f}\n")
            log_file.write("\n")

            log_file.write("Request Processing Summary:\n")
            log_file.write(f"Total Requests: {total_requests}\n")
            log_file.write(f"High Priority Solved: {solved_requests['high']}\n")
            log_file.write(f"Low Priority Solved: {solved_requests['low']}\n")
            log_file.write(f"Average Processing Time: {average_time:.2f}s\n\n")

            log_file.write("Path Information:\n")
            log_file.write(f"{'Request ID':<20} {'Priority':<10} {'Path':<50}\n")
            log_file.write("-" * 80 + "\n")
            for path_info in paths_taken:
                log_file.write(f"{path_info['request_id'][-15:]:<20} "
                             f"{path_info['priority']:<10} "
                             f"{' -> '.join(map(str, path_info['path'])):<50}\n")
        
            log_file.write("\nTop 5 Feature Importance by Model:\n")
            for name, results in model_results.items():
                log_file.write(f"\n{name} Feature Importance:\n")
                top_features = results['feature_importance'].head()
                for _, row in top_features.iterrows():
                    log_file.write(f"{row['feature']}: {row['importance']:.4f}\n")
                    
    except IOError as e:
        logging.error(f"Error writing to enhanced log file: {e}")


class TimedBatchProcessor:
    def __init__(self, network, task_queue, batch_size=10, interval_seconds=2):
        self.network = network
        self.task_queue = task_queue
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        self.total_processed = 0
        self.processing_stats = {
            'batches_processed': 0,
            'total_high_priority': 0,
            'total_low_priority': 0,
            'avg_processing_time': 0,
            'resource_utilization': [],
            'throughput': [],
            'high_priority_cpu_usage': [],
            'low_priority_cpu_usage': []
        }
    
    def process_batch(self, data_points, queue_decision_model):
        batch_start_time = time.time()
        
        for data_point in data_points:
            self.task_queue.add_to_queue(data_point, queue_decision_model)
        
        solved_requests, avg_time, paths = self.task_queue.solve_requests(self.network)
        
        self.total_processed += solved_requests['high'] + solved_requests['low']
        self.processing_stats['batches_processed'] += 1
        self.processing_stats['total_high_priority'] += solved_requests['high']
        self.processing_stats['total_low_priority'] += solved_requests['low']
        
        cpu_utilization, high_cpu, low_cpu = self._calculate_resource_utilization()
        self.processing_stats['resource_utilization'].append(cpu_utilization)
        self.processing_stats['high_priority_cpu_usage'].append(high_cpu)
        self.processing_stats['low_priority_cpu_usage'].append(low_cpu)
        
        batch_time = time.time() - batch_start_time
        throughput = (solved_requests['high'] + solved_requests['low']) / batch_time
        self.processing_stats['throughput'].append(throughput)
        
        return solved_requests, avg_time, paths
    
    def process_all_data(self, data_points, queue_decision_model):
        results = []
        total_points = len(data_points)
        processed = 0
        
        while processed < total_points:
            batch = data_points[processed:processed + self.batch_size]
            batch_results = self.process_batch(batch, queue_decision_model)
            results.append(batch_results)
            processed += self.batch_size
            time.sleep(self.interval_seconds)
        
        return results

    def _calculate_resource_utilization(self):
        total_cpu = 0
        used_cpu = 0
        high_priority_cpu = 0
        low_priority_cpu = 0
        
        MAX_CPU_PER_NODE = 3000
        
        for node in self.network.nodes():
            initial_cpu = MAX_CPU_PER_NODE
            current_cpu = min(float(self.network.nodes[node]['CPU']), initial_cpu)
            node_used_cpu = max(0, initial_cpu - current_cpu)
            
            total_cpu += initial_cpu
            used_cpu += node_used_cpu
            
            if self.total_processed > 0:
                high_ratio = min(1.0, max(0.0, self.processing_stats['total_high_priority'] / self.total_processed))
                node_high_cpu = node_used_cpu * high_ratio
                node_low_cpu = node_used_cpu * (1.0 - high_ratio)
                
                high_priority_cpu += node_high_cpu
                low_priority_cpu += node_low_cpu
        
        total_utilization = min(100.0, (used_cpu / total_cpu * 100.0)) if total_cpu > 0 else 0.0
        high_priority_pct = min(100.0, (high_priority_cpu / total_cpu * 100.0)) if total_cpu > 0 else 0.0
        low_priority_pct = min(100.0, (low_priority_cpu / total_cpu * 100.0)) if total_cpu > 0 else 0.0
        
        return total_utilization, high_priority_pct, low_priority_pct

def create_analysis_plots(batch_sizes=range(100, 1200, 100)):
    """analyis graphs and such"""
    output_dir = 'output_graphs'
    os.makedirs(output_dir, exist_ok=True)
    

    fig_loss = plt.figure(figsize=(12, 8))
    fig_metrics = plt.figure(figsize=(15, 10))
    

    gs_metrics = fig_metrics.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    all_results = []
    model_metrics = {
        'training_loss': [],
        'validation_loss': [],
        'training_accuracy': [],
        'validation_accuracy': [],
        'epochs': []
    }
    throughput_data = []
    resource_utilization = {
        'high_priority_cpu': [],
        'low_priority_cpu': [],
        'high_priority_bandwidth': [],
        'low_priority_bandwidth': []
    }
    
    for batch_size in batch_sizes:
        try:
            log_filename = f'enhanced_request_log_{batch_size}.txt'
            results = analyze_request_processing(log_filename)
            if results:
                results['batch_size'] = batch_size
                all_results.append(results)
                
                processed_requests = float(results['total_solved'])
                time_taken = float(results.get('processing_time', 60.0))
                throughput = processed_requests / time_taken
                throughput_data.append(throughput)
                
                high_priority_ratio = float(results['high_priority_solved']) / float(results['total_solved'])
                
                resource_utilization['high_priority_cpu'].append(high_priority_ratio * results.get('cpu_utilization', 100))
                resource_utilization['low_priority_cpu'].append((1 - high_priority_ratio) * results.get('cpu_utilization', 100))
                

                resource_utilization['high_priority_bandwidth'].append(high_priority_ratio * results.get('bandwidth_utilization', 100))
                resource_utilization['low_priority_bandwidth'].append((1 - high_priority_ratio) * results.get('bandwidth_utilization', 100))

                if 'model_metrics' in results:
                    for metric in results['model_metrics']:
                        model_metrics['training_loss'].append(metric.get('train_loss', 0))
                        model_metrics['validation_loss'].append(metric.get('val_loss', 0))
                        model_metrics['training_accuracy'].append(metric.get('train_acc', 0))
                        model_metrics['validation_accuracy'].append(metric.get('val_acc', 0))
                        model_metrics['epochs'].append(metric.get('epoch', 0))
                        
        except Exception as e:
            logging.warning(f"Skipping batch size {batch_size}: {str(e)}")
            continue
    
    plt.figure(fig_loss.number)
    gs_loss = fig_loss.add_gridspec(2, 1, hspace=0.3)
    
    ax_loss = fig_loss.add_subplot(gs_loss[0])
    ax_loss.plot(model_metrics['epochs'], model_metrics['training_loss'], 
                label='Training Loss', color='#2ecc71')
    ax_loss.plot(model_metrics['epochs'], model_metrics['validation_loss'], 
                label='Validation Loss', color='#e74c3c')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Model Training and Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    ax_acc = fig_loss.add_subplot(gs_loss[1])
    ax_acc.plot(model_metrics['epochs'], model_metrics['training_accuracy'], 
                label='Training Accuracy', color='#2ecc71')
    ax_acc.plot(model_metrics['epochs'], model_metrics['validation_accuracy'], 
                label='Validation Accuracy', color='#e74c3c')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Model Training and Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)
    
    plt.figure(fig_metrics.number)
    
    ax_throughput = fig_metrics.add_subplot(gs_metrics[0, 0])
    ax_throughput.plot(batch_sizes, throughput_data, marker='o', color='#3498db', linewidth=2)
    ax_throughput.set_xlabel('Batch Size')
    ax_throughput.set_ylabel('Requests/Second')
    ax_throughput.set_title('System Throughput')
    ax_throughput.grid(True)
    ax_cpu = fig_metrics.add_subplot(gs_metrics[0, 1])
    ax_cpu.stackplot(batch_sizes,
                    [resource_utilization['high_priority_cpu'],
                     resource_utilization['low_priority_cpu']],
                    labels=['High Priority', 'Low Priority'],
                    colors=['#2ecc71', '#e74c3c'])
    ax_cpu.set_xlabel('Batch Size')
    ax_cpu.set_ylabel('CPU Utilization (%)')
    ax_cpu.set_title('CPU Resource Utilization by Priority')
    ax_cpu.legend()
    ax_cpu.grid(True)
    ax_bw = fig_metrics.add_subplot(gs_metrics[1, 0])
    ax_bw.stackplot(batch_sizes,
                   [resource_utilization['high_priority_bandwidth'],
                    resource_utilization['low_priority_bandwidth']],
                   labels=['High Priority', 'Low Priority'],
                   colors=['#2ecc71', '#e74c3c'])
    ax_bw.set_xlabel('Batch Size')
    ax_bw.set_ylabel('Bandwidth Utilization (%)')
    ax_bw.set_title('Bandwidth Resource Utilization by Priority')
    ax_bw.legend()
    ax_bw.grid(True)
    ax_efficiency = fig_metrics.add_subplot(gs_metrics[1, 1])
    high_priority_total = np.array(resource_utilization['high_priority_cpu']) + \
                         np.array(resource_utilization['high_priority_bandwidth'])
    low_priority_total = np.array(resource_utilization['low_priority_cpu']) + \
                        np.array(resource_utilization['low_priority_bandwidth'])
    efficiency_ratio = high_priority_total / (high_priority_total + low_priority_total) * 100
    
    ax_efficiency.plot(batch_sizes, efficiency_ratio, marker='o', color='#9b59b6', linewidth=2)
    ax_efficiency.set_xlabel('Batch Size')
    ax_efficiency.set_ylabel('Resource Efficiency Ratio (%)')
    ax_efficiency.set_title('High Priority vs Low Priority Resource Efficiency')
    ax_efficiency.grid(True)
    fig_loss.tight_layout()
    fig_metrics.tight_layout()
    
    fig_loss.savefig(os.path.join(output_dir, 'model_training_metrics.png'), dpi=300, bbox_inches='tight')
    fig_metrics.savefig(os.path.join(output_dir, 'system_performance_metrics.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    #summary DataFrame
    summary_df = pd.DataFrame({
        'Batch Size': batch_sizes,
        'Throughput (req/s)': throughput_data,
        'High Priority CPU (%)': resource_utilization['high_priority_cpu'],
        'Low Priority CPU (%)': resource_utilization['low_priority_cpu'],
        'High Priority Bandwidth (%)': resource_utilization['high_priority_bandwidth'],
        'Low Priority Bandwidth (%)': resource_utilization['low_priority_bandwidth'],
        'Resource Efficiency Ratio (%)': efficiency_ratio
    })
    
    return summary_df, pd.DataFrame(all_results)

def analyze_request_processing(log_filename):
    with open(log_filename, 'r') as file:
        data = file.readlines()
        
    results = {
        'processed': [],
        'unprocessed': [],
        'reasons': [],
        'timestamps': [],
        'model_metrics': {}
    }
    model_section_start = data.index("Model Comparison Results:\n")
    model_metrics = {}
    
    for line in data[model_section_start + 2:]: 
        if line.strip() and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 4: 
                model_name = parts[0]
                model_metrics[model_name] = {
                    'accuracy': float(parts[1]),
                    'auc_roc': float(parts[2]),
                    'avg_precision': float(parts[3])
                }
        if line.startswith('\n'): 
            break
    
    for line in data:
        if 'Total Requests:' in line:
            total_requests = int(line.split(': ')[1])
        elif 'High Priority Solved:' in line:
            high_priority_solved = int(line.split(': ')[1])
        elif 'Low Priority Solved:' in line:
            low_priority_solved = int(line.split(': ')[1])
    
    total_solved = high_priority_solved + low_priority_solved
    unprocessed = total_requests - total_solved
    
    return {
        'total_requests': total_requests,
        'total_solved': total_solved,
        'unprocessed': unprocessed,
        'high_priority_solved': high_priority_solved,
        'low_priority_solved': low_priority_solved,
        'model_metrics': model_metrics
    }


def main():
    setup_logging()
    logging.info("Starting ECG data processing and analysis...")
    training_data_file = 'training_data_points.json'
    new_data_file = 'new_data_points.json'
    
    try:
        criticalness_model = CriticalnessModel()
        logging.info("Initialized CriticalnessModel")
        logging.info("Generating training data...")
        features, targets = generate_and_prepare_training_data(criticalness_model, training_data_file)
        logging.info(f"Generated training data with {len(features)} samples")
        logging.info("Training models...")
        model_results, X_test, y_test = train_models(features, targets)
        best_model = max(model_results.items(), key=lambda x: x[1]['auc_roc'])[1]['model']
        logging.info("Models trained successfully")
        batch_sizes = range(100, 1200, 100)  
        for total_requests in batch_sizes:
            logging.info(f"\nProcessing batch size: {total_requests}")
            network = network_creation()
            task_queue = TaskQueue()
            batch_processor = TimedBatchProcessor(
                network=network,
                task_queue=task_queue,
                batch_size=50,
                interval_seconds=2  
            )
            logging.info(f"Generating {total_requests} data points...")
            data_points = generate_data_points(total_requests, new_data_file, criticalness_model)
            
            logging.info("Processing data points...")
            all_results = batch_processor.process_all_data(data_points, best_model)
            total_solved = {
                'high': sum(r[0]['high'] for r in all_results),
                'low': sum(r[0]['low'] for r in all_results)
            }
            avg_time = sum(r[1] for r in all_results) / len(all_results)
            all_paths = [p for r in all_results for p in r[2]]
            log_filename = f'enhanced_request_log_{total_requests}.txt'
            log_results(
                log_filename, 
                total_requests, 
                total_solved, 
                avg_time, 
                all_paths, 
                network, 
                model_results
            )
            
            logging.info(f"Batch {total_requests} complete - "
                        f"High priority: {total_solved['high']}, "
                        f"Low priority: {total_solved['low']}")
            del data_points, all_results
            gc.collect()
        logging.info("\nGenerating analysis plots and summary...")
        summary_df = create_analysis_plots(batch_sizes)
        summary_df, all_results_df = create_analysis_plots(batch_sizes)
        print("\nProcessing Summary:")
        print(summary_df)
        print("\nSummary Statistics:")
        print(summary_df.describe()) 
        
        logging.info("Analysis complete. Check 'output_graphs' directory for results.")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
