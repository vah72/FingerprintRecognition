# Generated by Django 2.2 on 2023-01-01 13:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('manageData', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='student',
            name='label',
        ),
        migrations.AlterField(
            model_name='student',
            name='dob',
            field=models.DateField(),
        ),
    ]
