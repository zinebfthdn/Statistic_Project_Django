from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [    
              
   path('login/', views.user_login, name='login'),
   path('logout/', views.user_logout, name='logout'),
   path('', views.index, name='app'),
   path('excel/', views.excel, name='excel'), 
   path('CSV/', views.csv, name='csv'),
  
   path('visualiser/', views.visualiser, name='visualiser'),
   path('visualiserChart/', views.visualiser_chart, name='visualiser_chart'),
   path('diagramme/', views.diagramme, name='diagramme'),
   path('parcourir/', views.parcourir_chart, name='parcourir_chart'),
   path('binomiale/', views.Binomiale, name='Binomiale'),
   path('bernoulli/', views.Bernoulli, name='Bernoulli'),
   path('normale/', views.Normale, name='Normale'),
   path('poisson/', views.Poisson, name='Poisson'),
   path('uniforme/', views.Uniforme, name='Uniforme'),
   path('exponentielle/', views.Exponentielle, name='Exponentielle'),

   path('calcules/', views.Calcules, name='Calcules'),
   path('inferentielles/', views.inferentielles, name='inferentielles'),
    path('test_traitement/', views.test_traitement, name='test_traitement'),
]