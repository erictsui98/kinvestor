from django.urls import path
from hedge import views
from .views import PostListView, PostDetailView, PostCreateView, PostUpdateView, PostDeleteView, BasicUpdateView

urlpatterns = [
    path("", PostListView.as_view(), name="hedge-home"),
    path("post/<int:pk>/", PostDetailView.as_view(), name="post-detail"), #specify a specific post by provide the integer primary key of the post
    #var call pk because it is what the detaiView expected.
    path("post/new/", PostCreateView.as_view(), name="post-create"),
    path("post/<int:pk>/update/", PostUpdateView.as_view(), name="post-update"),
    #provide a url pattern, let the post update view handle the route, provide a name
    path("post/<int:pk>/delete/", PostDeleteView.as_view(), name="post-delete"),
    path("about/", views.about, name="hedge-about"),

    path("info/<int:pk>/update/", BasicUpdateView.as_view(), name="info-update"),
]