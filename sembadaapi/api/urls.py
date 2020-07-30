from django.conf.urls import url

from .detectwidget import (
    DetectWidgetAPIView,
)

from .getmodel import (
    GetModelAPIView,
)


urlpatterns = [
    url(r'^$', DetectWidgetAPIView.as_view()),
    url(r'^model/', GetModelAPIView.as_view()),
]
