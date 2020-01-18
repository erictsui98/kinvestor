from django.db import models
from django.utils import timezone #default
from django.contrib.auth.models import User
from django.urls import reverse


# can link to user if you want multiple set of stocks
class Post(models.Model):
	ticker = models.CharField(max_length=10)
	expected_return = models.FloatField(default=0.1)
    # content = models.TextField() #lines and lines of text
    # dataposted = models.DateTimeField(default = timezone.now) #auto_now and auto_now_add add=current time when created
    # author = models.ForeignKey(User, on_delete=models.CASCADE) #one user to many posts one-to-many foreign key

	def __str__(self):
		return self.ticker

	def get_absolute_url(self):
		return reverse('hedge-home')
    #specific post with a primary key pk value = self.pk

# one instance only, can link to user if you want multiple
class Basic(models.Model):
	duration = models.FloatField(default=1)
	target_return = models.FloatField(default=0.27)
	def get_absolute_url(self):
		return reverse('hedge-home')
