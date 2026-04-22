import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.shortcuts import render
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def index(request):
    return render(request, 'project1/index.html')


def upload(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)


        features = df.columns[:-1].tolist()
        label = df.columns[-1]


        if df[label].nunique() <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'


        request.session['dataset'] = df.to_json()
        request.session['problem_type'] = problem_type


        # Generate scatter plot
        fig, ax = plt.subplots(figsize=(7, 5))
        if problem_type == 'classification':
            for cls in df[label].unique():
                subset = df[df[label] == cls]
                ax.scatter(subset[features[0]], subset[features[1]], label=str(cls))
            ax.legend(title=label)
            ax.set_ylabel(features[1])
        else:
            ax.scatter(df[features[0]], df[label])
            ax.set_ylabel(label)


        ax.set_xlabel(features[0])
        ax.set_title('Data Visualization')


        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()


        context = {
            'features': features,
            'label': label,
            'problem_type': problem_type,
            'n_samples': len(df),
            'plot': plot_b64,
        }
        return render(request, 'project1/results.html', context)


    return render(request, 'project1/index.html', {'error': 'Please upload a CSV file.'})




def train(request):
    if request.method == 'POST':
        df = pd.read_json(request.session.get('dataset'))
        problem_type = request.session.get('problem_type', 'classification')


        features = df.columns[:-1].tolist()
        label = df.columns[-1]


        X = df[features].values
        y = df[label].values


        model_name  = request.POST.get('model_name', 'knn')
        split       = int(request.POST.get('split', 80)) / 100
        hyperparam  = int(request.POST.get('hyperparam', 5))


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split, random_state=42
        )


        # Pick model
        if model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=hyperparam)
        elif model_name == 'tree' and problem_type == 'classification':
            model = DecisionTreeClassifier(max_depth=hyperparam)
        elif model_name == 'tree' and problem_type == 'regression':
            model = DecisionTreeRegressor(max_depth=hyperparam)
        elif model_name == 'svm':
            model = SVC()
        else:
            model = LinearRegression()


        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        if problem_type == 'classification':
            score_name = 'Accuracy'
            score = round(accuracy_score(y_test, y_pred) * 100, 2)
        else:
            score_name = 'R² Score'
            score = round(r2_score(y_test, y_pred), 4)


        # Plot predictions vs actual
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)
        ax.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, marker='x')
        ax.legend()
        ax.set_title('Actual vs Predicted')
        ax.set_xlabel('Sample index')
        ax.set_ylabel(label)


        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pred_plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()


        context = {
            'model_name': model_name,
            'score_name': score_name,
            'score': score,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'pred_plot': pred_plot,
            'problem_type': problem_type,
        }
        return render(request, 'project1/train_results.html', context)


    return render(request, 'project1/index.html')


