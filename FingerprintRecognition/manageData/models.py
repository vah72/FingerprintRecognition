from django.db import models

# Create your models here.
class Employee(models.Model):
   
    name = models.CharField('Employee Name',null=False,max_length=255)
    dob = models.DateField('Date Of Birth')
    sample = models.IntegerField('Number of Sample', default=0, editable=False)
    def __str__(self) :
        return f"{self.name}"
class Image(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="training_data", editable=True)
    
    def __str__(self) :
        return f"{self.employee} - {self.image.url}"
   