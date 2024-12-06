
#updates from week 10
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
        # Increase initial CPU to 2-3 GHz (2000-3000 MHz)
        G.nodes[node]['CPU'] = float(random.uniform(2000, 3000))
    return G

def check_resources(network, path, functions_detail, bw, delay_threshold, queueing_delay, relaxation_factor=1.5):
    """
    More lenient resource checking with a relaxation factor
    """
    adjusted_delay_threshold = delay_threshold * relaxation_factor - queueing_delay
    calculated_delay = calculate_delays(functions_detail, len(path))
    
    # Slightly more permissive delay check
    if calculated_delay > adjusted_delay_threshold:
        logging.info(f"Path delay: {calculated_delay}, Threshold: {adjusted_delay_threshold}")
        return False
    
    for func in functions_detail:
        required_cpu = func['Required CPU']
        for node in path:
            # Allow small oversubscription of CPU
            if network.nodes[node]['CPU'] < required_cpu * relaxation_factor:
                logging.info(f"Node {node} CPU tight: Required {required_cpu}, Available {network.nodes[node]['CPU']}")
                return False
    
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        # Allow slight oversubscription of bandwidth
        if network.edges[edge]['BW'] < bw * relaxation_factor:
            logging.info(f"Bandwidth tight on edge {edge}: Required {bw}, Available {network.edges[edge]['BW']}")
            return False
    
    return True

def best_path(network, start, end, functions_detail, bw, delay_threshold, queueing_delay, max_paths=10):
    """
    More flexible path finding with multiple alternatives
    """
    try:
        # Find all simple paths, limit the number of paths to explore
        all_paths = list(nx.all_simple_paths(network, source=start, target=end, cutoff=None))[:max_paths]
        
        # Try multiple path relaxation strategies
        relaxation_factors = [1.0, 1.2, 1.5]
        for relaxation_factor in relaxation_factors:
            feasible_paths = [
                path for path in all_paths 
                if check_resources(
                    network, path, functions_detail, bw, 
                    delay_threshold, queueing_delay, 
                    relaxation_factor
                )
            ]
            
            if feasible_paths:
                return min(feasible_paths, key=lambda path: calculate_delays(functions_detail, len(path)))
        
        return None
    except Exception as e:
        logging.error(f"Error in best_path: {str(e)}")
        return None

def update_resources(network, path, functions_detail, bw, recovery_rate=0.1):
    """
    Updated resource update with partial recovery mechanism
    """
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        # Partial bandwidth recovery
        current_bw = network.edges[edge]['BW']
        recovery_amount = bw * recovery_rate
        network.edges[edge]['BW'] = float(min(current_bw + recovery_amount, 150))  # Set a max bandwidth
    
    for node in path:
        required_cpu = float(sum(func['Required CPU'] for func in functions_detail))
        current_cpu = network.nodes[node]['CPU']
        # Partial CPU recovery
        recovery_amount = required_cpu * recovery_rate
        network.nodes[node]['CPU'] = float(min(current_cpu + recovery_amount, 3000))  # Set a max CPU
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
        # Randomly choose a scenario
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
        # Extract features
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
        
        # Calculate priority score using multiple factors
        priority_score = (
            0.4 * avg_criticalness +  # Weight for criticalness
            0.3 * (anomalies / 10) +  # Normalized anomalies (assuming max 10)
            0.3 * (1 - min(transmission_delay / 500, 1))  # Normalized delay (inverse)
        )
        
        # Add some randomness to make it more realistic
        priority_score += random.uniform(-0.1, 0.1)
        
        # Convert to binary target (high priority = 1, low priority = 0)
        targets.append(1 if priority_score > 0.6 else 0)
    
    columns = list(feature_dict.keys())
    return pd.DataFrame(features, columns=columns), targets

def calculate_delays(functions_detail, path_length):
    processing_delay = float(sum(func['Required CPU'] for func in functions_detail))
    link_delay = float(path_length)
    forwarding_delay = float(path_length - 1) * 0.05 #forwarding delay reduced
    total_delay = processing_delay + link_delay + forwarding_delay
    return float(total_delay)


class TaskQueue:
    def __init__(self):
        # Single queue for all requests
        self.queue = []
        # Initialize thread lock
        self.lock = threading.Lock()
        # Counters for statistics
        self.stats = {
            'high_priority_added': 0,
            'low_priority_added': 0,
            'high_priority_processed': 0,
            'low_priority_processed': 0
        }

    def calculate_priority_score(self, remaining_time, is_high_priority, waiting_time):
        """
        Calculate a priority score that considers:
        - Remaining time before deadline
        - Priority level
        - Time spent waiting in queue
        With adjusted balancing to prevent high priority dominance
        """
        time_urgency = 1.0 / max(remaining_time, 1)  # Avoid division by zero
        waiting_factor = waiting_time / 1000  # Convert to seconds and normalize
        
        # Reduce the priority difference between high and low
        priority_factor = 1.2 if is_high_priority else 1.0
        
        # Give more weight to waiting time to prevent starvation
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
        # Calculate priority score for new request
        waiting_time = (datetime.now() - request_info['entry_time']).total_seconds() * 1000
        new_score = self.calculate_priority_score(
            request_info['remaining_time'],
            request_info['is_high_priority'],
            waiting_time
        )
        
        # Insert maintaining sort order by priority score
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
        
        # Track consecutive high priority solves to enforce fairness
        consecutive_high_priority = 0
        max_consecutive_high_priority = 3  # Maximum allowed consecutive high priority solves
        
        while True:
            with self.lock:
                if not self.queue:
                    break
                
                # Find the next appropriate request to process
                request_index = 0
                request_info = None
                
                # If we've processed too many high priority consecutively,
                # try to find a low priority request
                if consecutive_high_priority >= max_consecutive_high_priority:
                    for i, req in enumerate(self.queue):
                        if not req['is_high_priority']:
                            request_index = i
                            request_info = req
                            break
                
                # If no low priority found or no consecutive limit hit,
                # take the next request
                if request_info is None and self.queue:
                    request_info = self.queue[0]
                    request_index = 0
                
                if request_info is None:
                    break
                    
                # Remove the selected request from queue
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
                    
                    # Update statistics and consecutive counter
                    if is_high_priority:
                        solved_requests['high'] += 1
                        self.stats['high_priority_processed'] += 1
                        consecutive_high_priority += 1
                    else:
                        solved_requests['low'] += 1
                        self.stats['low_priority_processed'] += 1
                        consecutive_high_priority = 0  # Reset counter
                    
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

#Criticalness Model for Each Function 1-3 low, medium, high criticality
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
                # Default to a medium criticality for any unrecognized task
                criticality.append(2)

        return criticality


def generate_and_prepare_training_data(criticalness_model, training_data_file):
    generate_data_points(1000, training_data_file, criticalness_model)
    training_data_points = load_data_points(training_data_file)
    return extract_features_and_targets(training_data_points)

#random forest model, XGBoost, LightGBM
def train_models(features, targets):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # Initialize models
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
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Store results
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
            # Write header
            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"Enhanced Processing Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*50}\n\n")
            
            # Write model comparison results
            log_file.write("Model Comparison Results:\n")
            log_file.write(f"{'Model':<15} {'Accuracy':<10} {'AUC-ROC':<10} {'Avg Precision':<15}\n")
            log_file.write("-" * 50 + "\n")
            for name, results in model_results.items():
                log_file.write(f"{name:<15} {results['accuracy']:.4f} {results['auc_roc']:.4f} {results['avg_precision']:.4f}\n")
            log_file.write("\n")
            
            # Write summary statistics
            log_file.write("Request Processing Summary:\n")
            log_file.write(f"Total Requests: {total_requests}\n")
            log_file.write(f"High Priority Solved: {solved_requests['high']}\n")
            log_file.write(f"Low Priority Solved: {solved_requests['low']}\n")
            log_file.write(f"Average Processing Time: {average_time:.2f}s\n\n")
            
            # Write path information
            log_file.write("Path Information:\n")
            log_file.write(f"{'Request ID':<20} {'Priority':<10} {'Path':<50}\n")
            log_file.write("-" * 80 + "\n")
            for path_info in paths_taken:
                log_file.write(f"{path_info['request_id'][-15:]:<20} "
                             f"{path_info['priority']:<10} "
                             f"{' -> '.join(map(str, path_info['path'])):<50}\n")
            

            
            # Write feature importance for each model
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
            'resource_utilization': []
        }
    
    def process_batch(self, data_points, queue_decision_model):
        """Process a small batch of requests"""
        batch_start_time = time.time()
        
        # Add batch to queue
        for data_point in data_points:
            self.task_queue.add_to_queue(data_point, queue_decision_model)
        
        # Process current queue state
        solved_requests, avg_time, paths = self.task_queue.solve_requests(self.network)
        
        # Update statistics
        self.total_processed += solved_requests['high'] + solved_requests['low']
        self.processing_stats['batches_processed'] += 1
        self.processing_stats['total_high_priority'] += solved_requests['high']
        self.processing_stats['total_low_priority'] += solved_requests['low']
        
        # Calculate resource utilization
        cpu_utilization = self._calculate_resource_utilization()
        self.processing_stats['resource_utilization'].append(cpu_utilization)
        
        return solved_requests, avg_time, paths
    
    def _calculate_resource_utilization(self):
        """Calculate current network resource utilization"""
        total_cpu = 0
        used_cpu = 0
        for node in self.network.nodes():
            initial_cpu = float(random.randint(1000, 1500))  # Same as in network_creation
            current_cpu = float(self.network.nodes[node]['CPU'])
            total_cpu += initial_cpu
            used_cpu += (initial_cpu - current_cpu)
        
        return (used_cpu / total_cpu) * 100 if total_cpu > 0 else 0
    
    def process_all_data(self, data_points, queue_decision_model):
        """Process all data points in small batches with frequent network resource refresh"""
        total_batches = len(data_points) // self.batch_size + (1 if len(data_points) % self.batch_size else 0)
        all_results = []
    
        start_time = time.time()
        last_network_refresh = start_time
    
        for i in range(0, len(data_points), self.batch_size):
            current_time = time.time()
        
            # Check if network resources need to be refreshed (every 2-3 seconds)
            if current_time - last_network_refresh >= 2:  # You can adjust this threshold
                self.network = network_creation()  # Refresh network state
                last_network_refresh = current_time
        
            batch = data_points[i:i + self.batch_size]
        
            # Process batch
            results = self.process_batch(batch, queue_decision_model)
            all_results.append(results)
        
            # Log progress
            progress = (i + len(batch)) / len(data_points) * 100
            logging.info(f"Progress: {progress:.1f}% - Batch {i//self.batch_size + 1}/{total_batches}")
        
            # Wait for the specified interval before next batch
            if i + self.batch_size < len(data_points):
                time.sleep(self.interval_seconds)
    
        return all_results

def main():
    setup_logging()
    training_data_file = 'training_data_points.json'
    new_data_file = 'new_data_points.json'
    criticalness_model = CriticalnessModel()
    
    try:
        # Generate and prepare training data
        features, targets = generate_and_prepare_training_data(criticalness_model, training_data_file)
        
        # Train multiple models
        model_results, X_test, y_test = train_models(features, targets)
        best_model = max(model_results.items(), key=lambda x: x[1]['auc_roc'])[1]['model']
        
        # Process different batch sizes with the new timed processor
        for total_requests in range(100, 5000, 100):
            network = network_creation()
            task_queue = TaskQueue()
            
            # Configure batch processor with smaller batch sizes and intervals
            batch_processor = TimedBatchProcessor(
                network=network,
                task_queue=task_queue,
                batch_size=50,  # Process 20 requests at a time
                interval_seconds=3  # Wait 2 seconds between batches
            )
            
            # Generate and process data points
            data_points = generate_data_points(total_requests, new_data_file, criticalness_model)
            all_results = batch_processor.process_all_data(data_points, best_model)
            
            # Aggregate results
            total_solved = {
                'high': sum(r[0]['high'] for r in all_results),
                'low': sum(r[0]['low'] for r in all_results)
            }
            avg_time = sum(r[1] for r in all_results) / len(all_results)
            all_paths = [p for r in all_results for p in r[2]]
            
            # Log results
            log_filename = f'enhanced_request_log_{total_requests}.txt'
            log_results(log_filename, total_requests, total_solved, 
                       avg_time, all_paths, network, model_results)
            
            logging.info(f"Batch size {total_requests} complete: "
                        f"High priority: {total_solved['high']}, "
                        f"Low priority: {total_solved['low']}, "
                        f"Average utilization: {sum(batch_processor.processing_stats['resource_utilization'])/len(batch_processor.processing_stats['resource_utilization']):.2f}%")
            
    except Exception as e:
        logging.error(f"Error in enhanced main function: {str(e)}")

def analyze_request_processing(log_filename):
    results = {
        'processed': [],
        'unprocessed': [],
        'reasons': [],
        'timestamps': [],
        'model_metrics': {}
    }
    
    with open(log_filename, 'r') as file:
        data = file.readlines()
    
    # Extract model comparison results
    model_section_start = data.index("Model Comparison Results:\n")
    model_metrics = {}
    
    for line in data[model_section_start + 2:]:  # Skip header lines
        if line.strip() and not line.startswith('-'):
            parts = line.split()
            if len(parts) >= 4:  # Make sure we have all metrics
                model_name = parts[0]
                model_metrics[model_name] = {
                    'accuracy': float(parts[1]),
                    'auc_roc': float(parts[2]),
                    'avg_precision': float(parts[3])
                }
        if line.startswith('\n'):  # End of model section
            break
    
    # Extract processing statistics
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

def create_analysis_plots(batch_sizes=range(100, 1200, 100)):
    plt.style.use('seaborn')
    
    # Prepare data containers
    all_results = []
    model_performance = {
        'RandomForest': {'accuracy': [], 'auc_roc': [], 'avg_precision': []},
        'XGBoost': {'accuracy': [], 'auc_roc': [], 'avg_precision': []},
        'GradientBoosting': {'accuracy': [], 'auc_roc': [], 'avg_precision': []}
    }
    
    # Analyze each batch
    for batch_size in batch_sizes:
        try:
            log_filename = f'enhanced_request_log_{batch_size}.txt'
            results = analyze_request_processing(log_filename)
            results['batch_size'] = batch_size
            all_results.append(results)
            
            # Collect model metrics
            for model_name, metrics in results['model_metrics'].items():
                for metric_name, value in metrics.items():
                    model_performance[model_name][metric_name].append(value)
        except FileNotFoundError:
            print(f"Warning: Log file not found for batch size {batch_size}")
            continue
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Processed vs Unprocessed Requests
    ax1 = plt.subplot(321)
    batches = [r['batch_size'] for r in all_results]
    processed = [r['total_solved'] for r in all_results]
    unprocessed = [r['unprocessed'] for r in all_results]
    
    ax1.bar(batches, processed, label='Processed', color='green', alpha=0.6)
    ax1.bar(batches, unprocessed, bottom=processed, label='Unprocessed', color='red', alpha=0.6)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Number of Requests')
    ax1.set_title('Processed vs Unprocessed Requests')
    ax1.legend()
    
    # 2. Processing Success Rate
    ax2 = plt.subplot(322)
    success_rates = [r['total_solved']/r['total_requests']*100 for r in all_results]
    ax2.plot(batches, success_rates, marker='o', linewidth=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Request Processing Success Rate')
    
    # 3. Priority Distribution
    ax3 = plt.subplot(323)
    high_priority = [r['high_priority_solved'] for r in all_results]
    low_priority = [r['low_priority_solved'] for r in all_results]
    
    ax3.plot(batches, high_priority, label='High Priority', color='blue', marker='o')
    ax3.plot(batches, low_priority, label='Low Priority', color='orange', marker='o')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Number of Requests')
    ax3.set_title('Priority Distribution of Processed Requests')
    ax3.legend()
    
    # 4. Model Accuracy Comparison
    ax4 = plt.subplot(324)
    for model_name in model_performance.keys():
        ax4.plot(batches, model_performance[model_name]['accuracy'], 
                label=model_name, marker='o')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Model Accuracy Comparison')
    ax4.legend()
    
    # 5. Model AUC-ROC Comparison
    ax5 = plt.subplot(325)
    for model_name in model_performance.keys():
        ax5.plot(batches, model_performance[model_name]['auc_roc'], 
                label=model_name, marker='o')
    ax5.set_xlabel('Batch Size')
    ax5.set_ylabel('AUC-ROC')
    ax5.set_title('Model AUC-ROC Comparison')
    ax5.legend()
    
    # 6. Model Average Precision Comparison
    ax6 = plt.subplot(326)
    for model_name in model_performance.keys():
        ax6.plot(batches, model_performance[model_name]['avg_precision'], 
                label=model_name, marker='o')
    ax6.set_xlabel('Batch Size')
    ax6.set_ylabel('Average Precision')
    ax6.set_title('Model Average Precision Comparison')
    ax6.legend()
    
    output_dir = 'output_graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'request_analysis.png'),
        dpi=300,
        bbox_inches='tight',
        format='png',
        facecolor='white',
        edgecolor='none'
    )
    plt.close('all')
    gc.collect()  # Add memory cleanup
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Batch Size': result['batch_size'],
            'Total Requests': result['total_requests'],
            'Processed': result['total_solved'],
            'Unprocessed': result['unprocessed'],
            'Success Rate (%)': (result['total_solved']/result['total_requests']*100),
            'High Priority Processed': result['high_priority_solved'],
            'Low Priority Processed': result['low_priority_solved']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'processing_summary.csv'), index=False)
    
    return summary_df

if __name__ == "__main__":
    main()
    summary_df = create_analysis_plots()
    print("\nProcessing Summary:")
    print(summary_df)
    print("\nSummary Statistics:")
    print(summary_df.describe())
