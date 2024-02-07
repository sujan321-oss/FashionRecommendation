from django.contrib import admin

# Register your models here.
from .models import UploadImage

@admin.register(UploadImage)
class imagedata(admin.ModelAdmin):
    list_display=("id","image")
    
