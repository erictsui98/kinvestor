from django.urls import path

from . import views

urlpatterns = [
	path('', views.price),
    path('price/', views.price ,name='basic-price'),
    path('daily_return/', views.daily_return ,name='basic-daily_return'),
    path('past/avg_return/', views.avg_return ,name='past-avg_return'),
    path('past/estP/', views.pastEstP ,name='past-estP'),
    path('past/calP/', views.pastCalP ,name='past-calP'),
    path('past/PvsI/', views.pastPvsI ,name='past-PvsI'),
    path('exp/estP/', views.expEstP ,name='exp-estP'),
    path('exp/calP/', views.expCalP ,name='exp-calP'),
    path('exp/PvsI/', views.expPvsI ,name='exp-PvsI'),

    #path("refresh/", views.refresh, name="refresh"),

]
