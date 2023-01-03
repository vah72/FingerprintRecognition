from django import forms
from django.forms import *

from FingerprintRecognition.settings import DJANGO_LEDGER_FORM_INPUT_CLASSES
from django.utils.translation import gettext_lazy as _
from .models import Employee, Image
# Create your forms here.


# class FengyuanChenDatePickerInput(DateInput):
#     template_name = 'widgets/fengyuanchen_datepicker.html'
class NewEmployeeForm(ModelForm):
	class Meta :
		model = Employee
		fields = ['name', 'dob']
		widgets = {
		'name' : TextInput(attrs= {'class' : 'form-control',
                     'placeholder': _('Employee Name')}),
		'dob' : DateInput(attrs= {'class' : 'form-control', 
                           'placeholder': _('Date of Birtd (YYYY-MM-DD)')}),
		}
	# def clean(self):
	# 	super(NewStudentForm, self).clean()
	# 	name = self.cleaned_data['name']
	# 	dob = self.cleaned_data['dob']
	# 	inClass = self.cleaned_data['inCla']
	# 	if 
		
class chooseEmployeetForm(forms.Form):
    employee_id = IntegerField()

class showSampleForm(ModelForm):
    class Meta:
        model = Image
        fields = ['employee']
        labels = {'employee' : '',}

class ImageForm(ModelForm):
    image = ImageField(
		label="Image",
		widget= ClearableFileInput(attrs={"multiple" : True})
	)