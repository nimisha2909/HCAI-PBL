# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Nimisha Jethva", "matriculation": "670245"},
        {"name": "Prishma Dahal", "matriculation": "677158"},
        {"name": "Aishwarya Gosavi", "matriculation": "672418"},
    ]
    
    projects = [
        {"name": "project1", "url_name": "project1:index"}, 
    ]
    
    context = { 

        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))