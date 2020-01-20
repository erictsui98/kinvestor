from django.shortcuts import render 
from django.http import HttpResponse
from .models import Post, Basic
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
# import numpy as np
# import matplotlib.pyplot as plt, mpld3
import celery
app = celery.Celery('hedge')



@app.task
def home(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'hedge/home.html', context)
    #variable to loop through -> posts

class PostListView(ListView):
    model = Post
    template_name = 'hedge/home.html'
    #<app>/<model>_<viewtype>.html
    #hedge/post_list.html(default) -> specified the template to use -> home.html  
    #variable that default loop through -> object list
    context_object_name = 'posts'
    ordering = ['ticker']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['info'] = Basic.objects.first()
        return context


class PostDetailView(DetailView):
    model = Post
    #<app>/<model>_<viewtype>.html
    #hedge/post_detail.html


class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['ticker', 'expected_return']
    #<app>/<model>_<viewtype>.html
   
    def form_valid(self, form):  #solve integrity error
        # form.instance.author = self.request.user
        return super().form_valid(form)

class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['ticker', 'expected_return']
    #<app>/<model>_<viewtype>.html
   
    def form_valid(self, form):  #solve integrity error
        # form.instance.author = self.request.user
        return super().form_valid(form)

    #to prevent other user to update my post
    def test_func(self):
        # post = self.get_object()
        # #get_object = get the post that trying to update.
        # if self.request.user == post.author:
        #     return True
        # return False
        return True

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    #<app>/<model>_<viewtype>.html
    #hedge/post_detail.html

    #to prevent other user to update my post
    def test_func(self):
        post = self.get_object()
        #get_object = get the post that trying to update.
        # if self.request.user == post.author:
        #     return True
        # return False
        return True

    #redirect
    success_url = '/hedge'


def about(request):
    return render(request, 'hedge/about.html', {'title': 'About'})

class BasicUpdateView(UpdateView):
    model = Basic
    fields = ['duration', 'target_return']
    #<app>/<model>_<viewtype>.html
   
    def form_valid(self, form):  #solve integrity error
        # form.instance.author = self.request.user
        return super().form_valid(form)

#after create a view -> we need to create a url pattern in urls.py
#a tempalte too