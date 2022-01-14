from .models import *
from . import models
import sys
sys.modules['models'] = models

