"""FingerprintRecognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from manageData import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.homepage, name="homepage"),
    
      
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='home.html'), name='logout'),

    path('dashboard/', views.dashboard, name='dashboard'),
    path('train/', views.train, name='train'),
    path('choose_employee/', views.choose_employee, name='choose-employee'),
    path('show_sample/<employee_id', views.show_sample_employee, name='show-sample-employee'),
    path('view_statitics_home', views.view_statitics_home, name='view-statitics-home'),


    path('all_employees/', views.all_employee, name = 'list-employees'),
    path('register/', views.registryNewEml, name='register'),
    path('update_employee/<employee_id>', views.update_employee, name='update-employee'),
    path('delete_employee/<employee_id>', views.delete_employee, name='delete-employee'),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
