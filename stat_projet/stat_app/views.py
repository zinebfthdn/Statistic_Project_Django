from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import os
from django.conf import settings
from django.shortcuts import render, redirect, HttpResponse
import pandas as pd  # Note the corrected import statement
import requests
from .forms import BinomialForm,FileUploadForm,ExponentielleForm,TraitementForm,UniformeForm,PoissonForm,NormaleForm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64
import json
import plotly.express as px
import matplotlib
import plotly.graph_objs as go
from scipy.stats import binom
from django.http import JsonResponse
import plotly.io as pio
from .forms import BernoulliForm 
from scipy.stats import bernoulli,norm, t,expon, poisson,uniform
from scipy import stats
from unittest import result
matplotlib.use('Agg')
from statistics import mean, median, mode, variance, stdev
import plotly.figure_factory as ff
import io

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Associe l'utilisateur à la session
            return redirect('app')  # Redirige vers une vue protégée
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    return render(request, 'login.html')
def user_logout(request):
    logout(request)
    return redirect('login')  # Redirige vers la page de connexion après déconnexion

@login_required
def index(request):
    return render(request, 'index.html')



def generate_chart(df, type_chart, col1, col2):
    buffer = BytesIO()

    if type_chart == 'Barplot':
        fig = px.bar(df, x=col1, y=col2)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Bar Plot')
        return fig.to_json()

    elif type_chart == 'histogram':
        fig = px.histogram(df, x=col1)
        fig.update_layout(xaxis_title=col1, yaxis_title='Count', title='Histogram', barmode='overlay', bargap=0.1)
        return fig.to_json()

    elif type_chart == 'piechart':
        value_counts = df[col1].value_counts().reset_index()
        value_counts.columns = [col1, 'Count']
        fig = px.pie(value_counts, values='Count', names=col1, title='Pie Chart')
        return fig.to_json()


    elif type_chart == 'scatterplot':
        fig = px.scatter(df, x=col1, y=col2)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Scatter Plot')
        return fig.to_json()

    elif type_chart == 'heatmap':
        df_encoded = df.copy()
        for column in df_encoded.columns:
            if df_encoded[column].dtype == 'object':
                df_encoded[column], _ = pd.factorize(df_encoded[column])
        fig = px.imshow(df_encoded.corr(), color_continuous_scale='Viridis')
        fig.update_layout(title='Heatmap')
        return fig.to_json()

    elif type_chart == 'lineplot':
       
        fig = px.line(df, x=col1, y=col2,markers=True)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Line Plot')
        return fig.to_json()

        
    elif type_chart == 'boxplot':
        fig = px.box(df, x=col1)
        fig.update_layout(title='Box Plot')
        return fig.to_json()
        

    elif type_chart == 'violinplot':
        fig = px.violin(df, y=col1, box=True)
        fig.update_layout(yaxis_title=col1, title='Violin Plot')
        return fig.to_json()


  

    elif type_chart == 'kdeplot':
        data_to_plot = df[col1].replace([np.inf, -np.inf], np.nan).dropna()

        group_labels = ['distplot']  # This will be the label for the distribution
        # Generate the KDE plot with the histogram
        fig = ff.create_distplot([data_to_plot], group_labels, curve_type='kde', show_hist=True, histnorm='probability density')

        # Mise à jour de la disposition (layout) pour correspondre au style souhaité
        fig.update_layout(
            title="Kernel Density Estimation (KDE) Plot",
            yaxis_title="Density",
            xaxis_title=col1,
            showlegend=False,
            template='plotly_white'
        )

        # Update traces to match the desired style
        fig.update_traces(marker=dict(color='grey', line=dict(color='black', width=1.5)))

        fig_json = fig.to_json()

        return fig_json

@login_required
def csv(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            fichier = request.FILES['file']
            if fichier.name.endswith('.csv'):
                # Traitez le fichier CSV
                data = pd.read_csv(fichier)
                df = pd.DataFrame(data)
                columns_choices = [(col, col) for col in df.columns]
                df_json = df.to_json()
                request.session['df_json'] = df_json

                return render(
                        request,
                        'visualiser_data.html',
                        {'form': form,  'df': df.to_html(classes='table table-bordered'), 'column_names': df.columns},
                )
            else:
                return HttpResponse("Seuls les fichiers CSV sont autorisés. Veuillez télécharger un fichier CSV.")
    else:
        form = FileUploadForm()

    return render(request, 'csv.html', {'form': form})
@login_required
def excel(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)

        if form.is_valid():
            fichier = request.FILES['file']

            if fichier.name.endswith(('.xls', '.xlsx')):
                try:
                    data = pd.read_excel(fichier)
                    df = pd.DataFrame(data)
                    columns_choices = [(col, col) for col in df.columns]
                    df_json = df.to_json()
                    request.session['df_json']=df_json
                    return render(
                        request,
                        'visualiser_data.html',
                        {'form': form,  'df': df.to_html(classes='table table-bordered'), 'column_names': df.columns},
                    )
                except pd.errors.ParserError as e:
                    e = f"Erreur : Impossible de lire le fichier Excel. Assurez-vous que le fichier est au format Excel valide."
                    return render(request, 'excel.html', {'form': form, 'error_message': e})
            else:
                return HttpResponse("Seuls les fichiers Excel (.xls, .xlsx) sont autorisés. Veuillez télécharger un fichier Excel.")
    else:
        form = FileUploadForm()

    return render(request, 'excel.html', {'form': form})

def visualiser(request): 
    return render(request, 'visualiser_data.html')


@login_required
def visualiser_chart(request): 
    if request.method == 'POST':
        col1 = request.POST['col_name1']
        col2 = request.POST['col_name2']
        type_chart = request.POST['type_chart']
        df_json = request.session.get('df_json')
        
        df_json_io = StringIO(df_json)
        df = pd.read_json(df_json_io)
       
        if pd.api.types.is_string_dtype(df[col1]) and type_chart in ['kdeplot', 'violinplot', 'boxplot']:
            error_message = f"La colonne choisie est '{col1}'de type 'string', veuillez choisir une autre colonne."
            return render(request, 'diagramme.html', {'error_message': error_message})
        elif pd.api.types.is_string_dtype(df[col2]) and type_chart in ['Barplot', 'lineplot']:
            error_message = f"La deuxième colonne '{col2}' doit contenir des valeurs numériques. Veuillez choisir une autre colonne."
            return render(request, 'diagramme.html', {'error_message': error_message})
        elif type_chart == 'scatterplot':
            # Vérifier si l'une des colonnes n'est pas numérique
            if pd.api.types.is_string_dtype(df[col1]) or pd.api.types.is_string_dtype(df[col2]):
                # Préparer un message d'erreur en indiquant les noms des colonnes non numériques
                non_numeric_columns = [col for col in [col1, col2] if pd.api.types.is_string_dtype(df[col])]
                error_message = f"Les colonnes '{', '.join(non_numeric_columns)}' doivent être numériques. Veuillez choisir d'autres colonnes."
                return render(request, 'diagramme.html', {'error_message': error_message})

        elif type_chart=="Nothing":
            error_message = "Veuillez sélectionner un diagramme à afficher"
            return render(request, 'diagramme.html', {'error_message': error_message})
        
        chart = generate_chart(df, type_chart, col1, col2)
        return render(request, 'diagramme.html', {'chart': chart})
    
    return render(request, 'visualiser_data.html')

@login_required
def diagramme(request):
    return render(request, 'diagramme.html')



@login_required
def parcourir_chart(request):
    df = None
    columns_choices = None
    error_message = ""
    max_row = 0

    if 'df_json' in request.session:
        df_json = request.session['df_json']
        df = pd.read_json(StringIO(df_json))
        columns_choices = [col for col in df.columns]
        max_row = df.shape[0] - 1
        
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')
        parcourir_chart_type = request.POST.get('parcourir_chart')
        col_name1 = request.POST.get('col_name1')
        row_numb = request.POST.get('RowNumb')
        
        if selected_columns:
            df = df[selected_columns]

        if parcourir_chart_type == 'GroupBy':
            numeric_column = request.POST.get('numeric_column')
            condition = request.POST.get('condition')
            value = request.POST.get('value')

            if numeric_column and condition and value :
                try:
                    grouped_df = df.groupby(numeric_column)
                    value = float(value)
                    if condition == '>' :
                        df = grouped_df.filter(lambda x: x[numeric_column].mean() > value)
                    elif condition == '<':
                        df = grouped_df.filter(lambda x: x[numeric_column].mean() < value)
                    elif condition == '=':
                        df = grouped_df.filter(lambda x: x[numeric_column].mean() == value)
                except Exception as e:
                    error_message = f"Une erreur est survenue : {str(e)}"

            # Utilisez numeric_columns pour GroupBy dans le contexte
            contexte = {
                'df': df.to_html(classes='table table-bordered') if df is not None else None,
                'column_names': columns_choices,
                'max_row': max_row,
                'error_message': error_message
            }
            return render(request, 'parcourir.html', contexte)
        if parcourir_chart_type == 'FindElem' and df is not None:
            try:
                row_numb = int(row_numb)
                row_numb = min(row_numb, max_row)
                resultats_recherche = df.at[row_numb, col_name1]
                contexte = {'resultat': resultats_recherche, 'column_names': columns_choices, 'df': df.to_html(classes='table table-bordered'), 'max_row': max_row}
                return render(request, 'parcourir.html', contexte)
            except (ValueError, KeyError, IndexError):
                pass

        parcourir_rows_type = request.POST.get('parcourir_rows')

        if parcourir_rows_type == 'NbrOfRowsTop':
            nb_rows_top = int(request.POST.get('Head'))
            df = df.head(nb_rows_top)
        elif parcourir_rows_type == 'NbrOfRowsBottom':
            nb_rows_bottom = int(request.POST.get('Tail'))
            df = df.tail(nb_rows_bottom)
        elif parcourir_rows_type == 'FromRowToRow':
            from_row = int(request.POST.get('FromRowNumb'))
            to_row = int(request.POST.get('ToRowNumb'))
            df = df.loc[from_row:to_row]

    contexte = {
        'df': df.to_html(classes='table table-bordered') if df is not None else None,
        'column_names': columns_choices,  # Utilisez columns_choices ici pour les autres actions
        'max_row': max_row
    }   
    return render(request, 'parcourir.html', contexte)

#//////////////////////////////// LOIS ////////////////////////////////////////////////////////////////

@login_required
def Binomiale(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = BinomialForm(request.POST)
        if form.is_valid():
            n = form.cleaned_data['n']
            p = form.cleaned_data['p']

            # Générer des données de la distribution binomiale
            data_binom = binom.rvs(n=n, p=p, loc=0, size=1000)

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_binom, kde=True, stat='probability')
            ax.set(xlabel='Binomial', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = BinomialForm()

    return render(request, 'binomiale.html', {'form': form, 'plot_data': plot_data})

@login_required
def Bernoulli(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = BernoulliForm(request.POST)
        if form.is_valid():
            p = form.cleaned_data['p']

            # Générer des données de la distribution de Bernoulli
            data_bern = bernoulli.rvs(size=1000, p=p)

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_bern, kde=True, stat='probability')
            ax.set(xlabel='Bernoulli', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = BernoulliForm()

    return render(request, 'bernoulli.html', {'form': form, 'plot_data': plot_data})

@login_required
def Normale(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = NormaleForm(request.POST)
        if form.is_valid():
            mean = form.cleaned_data['mean']
            std_dev = form.cleaned_data['std_dev']

            # Points pour la courbe de la distribution normale
            x_values = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
            y_values = norm.pdf(x_values, mean, std_dev)

            # Créer la courbe de la distribution normale remplie avec Matplotlib
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            plt.fill_between(x_values, y_values, color="skyblue", alpha=0.4)
            plt.plot(x_values, y_values, color="Slateblue", alpha=0.6)
            plt.title('Distribution Normale Continue')
            plt.xlabel('Valeur')
            plt.ylabel('Densité de probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = NormaleForm()

    return render(request, 'normale.html', {'form': form, 'plot_data': plot_data})

@login_required
def Poisson(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = PoissonForm(request.POST)
        if form.is_valid():
            lambda_param = form.cleaned_data['lambda_param']

            # Générer des données de la distribution de Poisson
            data_poisson = poisson.rvs(mu=lambda_param, size=1000)

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_poisson, kde=True, stat='probability')
            ax.set(xlabel='Poisson', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = PoissonForm()

    return render(request, 'poisson.html', {'form': form, 'plot_data': plot_data})

@login_required
def Uniforme(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = UniformeForm(request.POST)
        if form.is_valid():
            a = form.cleaned_data['a']
            b = form.cleaned_data['b']

            # Générer des données de la distribution uniforme
            data_unif = uniform.rvs(loc=a, scale=b-a, size=1000)  # b = loc + scale

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_unif, kde=True, stat='probability')
            ax.set(xlabel='Uniforme', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = UniformeForm()

    return render(request, 'uniforme.html', {'form': form, 'plot_data': plot_data})


 # Assurez-vous d'importer votre formulaire ici

@login_required
def Exponentielle(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = ExponentielleForm(request.POST)
        if form.is_valid():
            beta = form.cleaned_data['beta']

            # Générer des échantillons de la distribution exponentielle
            data_exponentielle = expon.rvs(scale=beta, size=1000)

            # Créer la courbe de densité avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            sns.kdeplot(data_exponentielle, fill=True)
            plt.title('Distribution Exponentielle')
            plt.xlabel('Valeur')
            plt.ylabel('Densité de probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = ExponentielleForm()

    return render(request, 'exponentielle.html', {'form': form, 'plot_data': plot_data})

#/////////////////////////////////////////////////////////calcules

import numpy as np


def mode(valeurs):
    valeurs = np.array(valeurs.replace(';', ',').split(','), dtype=float)
    uniques, counts = np.unique(valeurs, return_counts=True)
    max_count = np.max(counts)
    modes = uniques[counts == max_count]
    if max_count == 1 or len(modes) == len(uniques):
        return "Pas de mode fréquent"
    else:
        return modes.tolist()

@login_required
def Calcules(request):
    if request.method == 'POST':
        form = TraitementForm(request.POST)
        if form.is_valid():
            valeurs_input = form.cleaned_data['valeurs']
            
            # Traiter les valeurs saisies
            valeurs = [float(x.strip()) for x in valeurs_input.replace(';', ',').split(',') if x.strip()]
            
            # Calcul des statistiques
            mean_value = np.mean(valeurs)
            median_value = np.median(valeurs)
            mode_value = mode(valeurs_input)
            variance_value = np.var(valeurs)
            stdev_value = np.std(valeurs)
            range_value = np.max(valeurs) - np.min(valeurs)

            return render(request, 'calcules.html', {'form': form, 'mean': mean_value,
                                                     'median': median_value, 'mode': mode_value,
                                                     'variance': variance_value, 'stdev': stdev_value, 'range': range_value })
    else:
        form = TraitementForm()

    return render(request, 'calcules.html', {'form': form})


#/////////////////////////////////////////////testes

def calculate_z_test(field, zTestmi, sigma, n, significance):
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)
    zTestmi = float(zTestmi.replace(',', '.'))
    z_stat = (field - zTestmi) / (sigma / np.sqrt(n))
    p_value_two_sided = norm.sf(abs(z_stat)) * 2
    hypothesis_result_two_sided = "On rejette l'hypothèse." if p_value_two_sided < significance else "On accepte l'hypothèse."
    return {
        'z_statistic': z_stat,
        'p_value_two_sided': p_value_two_sided,
        'hypothesis_result_two_sided': hypothesis_result_two_sided,
    }

def calculate_t_test2(field, tTestmi, sigma, n, significance):
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)
    tTestmi = float(tTestmi.replace(',', '.'))
    t_statistic = (field - tTestmi) / (sigma / np.sqrt(n))
    p_value_two_sided = t.sf(abs(t_statistic), df=n-1) * 2
    hypothesis_result_two_sided = "On rejette l'hypothèse." if p_value_two_sided < significance else "On accepte l'hypothèse."
    return {
        't_statistic': t_statistic,
        'p_value_two_sided': p_value_two_sided,
        'hypothesis_result_two_sided': hypothesis_result_two_sided,
    }


def test_traitement(request):
    if request.method == 'GET':
        test_type = request.GET.get('testType')
        if test_type:
            significance = float(request.GET.get('significance', 0.05))
            if test_type == 'zTest':
                field = request.GET.get('zTestField')
                sigma = request.GET.get('zTestSigma')
                n = request.GET.get('zTestN')
                zTestmi = request.GET.get('zTestmi')
                z_test_results = calculate_z_test(field, zTestmi, sigma, n, significance)
                return JsonResponse({
                    'z_statistic': z_test_results['z_statistic'],
                    'p_value_two_sided': z_test_results['p_value_two_sided'],
                    'hypothesis_result_two_sided': z_test_results['hypothesis_result_two_sided'],
                    'formule': "Z = (X̄ - μ) / (σ/ √n)"
                })
            elif test_type == 'tTest2':
                field = request.GET.get('tTestField2')
                sigma = request.GET.get('tTestSigma2')
                n = request.GET.get('testTestN2')
                tTestmi = request.GET.get('tTestmi2')
                t_test_results = calculate_t_test2(field, tTestmi, sigma, n, significance)
                return JsonResponse({
                    't_statistic': t_test_results['t_statistic'],
                    'p_value_two_sided': t_test_results['p_value_two_sided'],
                    'hypothesis_result_two_sided': t_test_results['hypothesis_result_two_sided'],
                    'formule': "Z = (X̄ - μ) / (σ/ √n)"
                })
            else:
                return JsonResponse({'error': 'Invalid test type'})
        else:
            return JsonResponse({'error': 'Invalid test type'})
    else:
        return JsonResponse({'error': 'Invalid request method'})

@login_required
def inferentielles(request):
    return render(request, 'inferentielles.html')