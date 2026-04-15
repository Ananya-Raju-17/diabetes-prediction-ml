import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cirq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import minimize
import time

class DiabetesClassifier:
    def __init__(self, data_path, quantum_sample_size=100):
        self.data_path = data_path
        self.quantum_sample_size = quantum_sample_size
        self.qubits = [cirq.GridQubit(0, i) for i in range(2)]
        
    def load_and_preprocess(self):
        print("Loading and preprocessing data...")
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=['Diabetes_012'])
        y = data['Diabetes_012']
        
        # Feature selection
        print("Performing feature selection with Lasso...")
        lasso = LassoCV(cv=5)
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        print(f"Selected features: {list(selected_features)}")
        
        X_selected = X[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)
        
    def run_classical_models(self):
        print("\nTraining classical models...")
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(penalty='l2'),
            "Decision Tree": DecisionTreeClassifier(min_samples_split=2),
            "Random Forest": RandomForestClassifier(n_estimators=100, bootstrap=True),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
        }
        
        self.classical_results = {}
        for name, model in models.items():
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            elapsed = time.time() - start_time
            
            self.classical_results[name] = {
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "Report": classification_report(self.y_test, y_pred),
                "Time": elapsed
            }
            print(f"{name} - Accuracy: {self.classical_results[name]['Accuracy']:.4f} (Time: {elapsed:.2f}s)")
    
    def prepare_quantum_data(self):
        # Reduce dataset size for quantum processing
        np.random.seed(42)
        indices = np.random.choice(len(self.X_train), self.quantum_sample_size, replace=False)
        self.X_train_q = self.X_train[indices, :2]  # Use first 2 features
        self.X_test_q = self.X_test[:self.quantum_sample_size, :2]
        self.y_train_q = self.y_train.iloc[indices]
        self.y_test_q = self.y_test.iloc[:self.quantum_sample_size]
    
    def quantum_feature_map(self, x, qubits):
        circuit = cirq.Circuit()
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(x[i] * np.pi).on(qubit))
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        return circuit
    
    def batched_quantum_kernel(self, X1, X2, batch_size=10):
        n1, n2 = len(X1), len(X2)
        kernel = np.zeros((n1, n2))
        
        for i in range(0, n1, batch_size):
            for j in range(0, n2, batch_size):
                X1_batch = X1[i:i+batch_size]
                X2_batch = X2[j:j+batch_size]
                batch_kernel = np.zeros((len(X1_batch), len(X2_batch)))
                
                for bi, x1 in enumerate(X1_batch):
                    for bj, x2 in enumerate(X2_batch):
                        circuit = self.quantum_feature_map(x1, self.qubits) + \
                                 self.quantum_feature_map(x2, self.qubits)
                        result = cirq.Simulator().simulate(circuit)
                        batch_kernel[bi, bj] = np.abs(np.sum(result.final_state_vector)) ** 2
                
                kernel[i:i+batch_size, j:j+batch_size] = batch_kernel
                print(f"Completed batch ({i//batch_size+1},{j//batch_size+1}) of ({(n1//batch_size)+1},{(n2//batch_size)+1})")
        
        return kernel
    
    def run_quantum_model(self):
        print("\nRunning quantum model...")
        self.prepare_quantum_data()
        
        print("Computing quantum kernel matrices...")
        start_time = time.time()
        K_train = self.batched_quantum_kernel(self.X_train_q, self.X_train_q)
        K_test = self.batched_quantum_kernel(self.X_test_q, self.X_train_q)
        
        print("Training Quantum SVM...")
        qsvm = SVC(kernel='precomputed')
        qsvm.fit(K_train, self.y_train_q)
        y_pred_q = qsvm.predict(K_test)
        
        self.quantum_results = {
            "Accuracy": accuracy_score(self.y_test_q, y_pred_q),
            "Report": classification_report(self.y_test_q, y_pred_q),
            "Time": time.time() - start_time
        }
        print(f"Quantum SVM - Accuracy: {self.quantum_results['Accuracy']:.4f} (Time: {self.quantum_results['Time']:.2f}s)")
    
    def visualize_results(self):
        model_names = list(self.classical_results.keys()) + ["Quantum SVM"]
        accuracies = [self.classical_results[name]["Accuracy"] for name in self.classical_results] + \
                    [self.quantum_results["Accuracy"]]
        times = [self.classical_results[name]["Time"] for name in self.classical_results] + \
                [self.quantum_results["Time"]]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy plot
        bars = ax1.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta'])
        ax1.set_title("Model Accuracy Comparison")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0.7, 0.9)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        # Time plot
        time_bars = ax2.bar(model_names, times, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta'])
        ax2.set_title("Training Time Comparison")
        ax2.set_ylabel("Time (seconds)")
        for bar in time_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results_comparison.png')
        print("\nResults visualization saved as 'results_comparison.png'")
    
    def save_results(self):
        with open('results_summary.txt', 'w') as f:
            f.write("Diabetes Classification Results\n")
            f.write("="*50 + "\n\n")
            
            f.write("Classical Models:\n")
            f.write("-"*50 + "\n")
            for name, res in self.classical_results.items():
                f.write(f"{name} - Accuracy: {res['Accuracy']:.4f} (Time: {res['Time']:.2f}s)\n")
                f.write(res['Report'] + "\n")
            
            f.write("\nQuantum Model (on subset):\n")
            f.write("-"*50 + "\n")
            f.write(f"Quantum SVM - Accuracy: {self.quantum_results['Accuracy']:.4f} (Time: {self.quantum_results['Time']:.2f}s)\n")
            f.write(f"Note: Quantum model trained on subset of {self.quantum_sample_size} samples\n")
            f.write(self.quantum_results['Report'])
        
        print("Results summary saved as 'results_summary.txt'")

def main():
    classifier = DiabetesClassifier(
        data_path='diabetes_health_indicators.csv',
        quantum_sample_size=100  # Adjust based on your system's memory
    )
    
    classifier.load_and_preprocess()
    classifier.run_classical_models()
    classifier.run_quantum_model()
    classifier.visualize_results()
    classifier.save_results()

if __name__ == "__main__":
    main()