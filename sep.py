from collections.abc import Iterable
from river.drift import PageHinkley

DIVISORS = {
	'pagehinkley': drift.PageHinkley,
	'ph': drift.PageHinkley,
}

class Separator():
  
  def __init__(self, divisor: str, divisor_params: Iterable):
    self.divisor = lower(divisor)
    
  
  def divide_time_series(self, series: Iterable):
    match	self.divisor:
      case 'ph' | 'pagehinkley':
        self.divide_page_hinkley(series)