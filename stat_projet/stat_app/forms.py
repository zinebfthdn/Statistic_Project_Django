from django import forms

class FileUploadForm(forms.Form):
    csv_file = forms.FileField(label='Fichier CSV', required=False)
    excel_file = forms.FileField(label='Fichier Excel', required=False)
    file = forms.FileField()

 
    # forms.py
class BinomialForm(forms.Form):
    n = forms.IntegerField(label='Nombre d\'essais', initial=10, min_value=1)
    p = forms.FloatField(label='Probabilité de succès', initial=0.5, min_value=0, max_value=1)

class BernoulliForm(forms.Form):
    p = forms.FloatField(label='Probabilité de succès', initial=0.5, min_value=0, max_value=1)
    
class NormaleForm(forms.Form):
    mean = forms.FloatField(label='Moyenne')
    std_dev = forms.FloatField(label='Écart-type')
    

class PoissonForm(forms.Form):
    lambda_param = forms.FloatField(label='Lambda (λ)')
    
class UniformeForm(forms.Form):
    a = forms.FloatField(label='a')
    b = forms.FloatField(label='b')

class ExponentielleForm(forms.Form):
    beta = forms.FloatField(label='Beta (β)')

class TraitementForm(forms.Form):
    valeurs = forms.CharField(label='Liste de valeurs', widget=forms.TextInput(attrs={'placeholder': 'Entrez les valeurs séparées par des tirets (-) ou des virgules (,)'}))

class VisualizationForm(forms.Form):
    CHART_CHOICES = [
        ('histplot', 'Histogramme'),
        ('scatterplot', 'Nuage de points'),
        ('barplot', 'Diagramme à barres'),
        ('heatmap', 'Carte de chaleur'),
        ('lineplot', 'Graphique linéaire'),
        ('boxplot', 'Boîte à moustaches'),
        ('histogram', 'Histogramme'),
        ('kdeplot', 'Graphique KDE'),
        ('violinplot', 'Violon'),
        ('piechart', 'Diagramme circulaire'),
    ]

    chart_type = forms.ChoiceField(choices=CHART_CHOICES, label='Type de Diagramme', required=False)
    column_name_1 = forms.ChoiceField(choices=[], label='Nom de la colonne 1', required=False)
    column_name_2 = forms.ChoiceField(choices=[], label='Nom de la colonne 2', required=False)
