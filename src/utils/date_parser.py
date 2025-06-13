"""Natural language date parsing utilities."""
from datetime import datetime, timedelta, date
from typing import Tuple, Optional
import re
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta


class DateParser:
    """Parse natural language dates into date ranges."""
    
    @staticmethod
    def parse_date_query(query: str, reference_date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
        """
        Parse natural language date query into start and end datetime.
        
        Args:
            query: Natural language date query
            reference_date: Reference date for relative queries (default: now)
            
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        query_lower = query.lower().strip()
        
        # Today
        if query_lower in ['today', 'now']:
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
            return start, end
        
        # Yesterday
        if query_lower == 'yesterday':
            yesterday = reference_date - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
            return start, end
        
        # This week
        if query_lower in ['this week', 'current week']:
            start = reference_date - timedelta(days=reference_date.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7) - timedelta(microseconds=1)
            return start, end
        
        # Last week
        if query_lower == 'last week':
            last_week = reference_date - timedelta(weeks=1)
            start = last_week - timedelta(days=last_week.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7) - timedelta(microseconds=1)
            return start, end
        
        # This month
        if query_lower in ['this month', 'current month']:
            start = reference_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = start + relativedelta(months=1)
            end = next_month - timedelta(microseconds=1)
            return start, end
        
        # Last month
        if query_lower == 'last month':
            last_month = reference_date - relativedelta(months=1)
            start = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = start + relativedelta(months=1)
            end = next_month - timedelta(microseconds=1)
            return start, end
        
        # Last N days/hours
        last_n_pattern = r'last (\d+) (day|days|hour|hours|week|weeks)'
        match = re.match(last_n_pattern, query_lower)
        if match:
            n = int(match.group(1))
            unit = match.group(2).rstrip('s')  # Remove plural 's'
            
            if unit == 'hour':
                start = reference_date - timedelta(hours=n)
            elif unit == 'day':
                start = reference_date - timedelta(days=n)
            elif unit == 'week':
                start = reference_date - timedelta(weeks=n)
            else:
                start = reference_date
            
            return start, reference_date
        
        # Past N days/hours
        past_n_pattern = r'past (\d+) (day|days|hour|hours|week|weeks)'
        match = re.match(past_n_pattern, query_lower)
        if match:
            n = int(match.group(1))
            unit = match.group(2).rstrip('s')
            
            if unit == 'hour':
                start = reference_date - timedelta(hours=n)
            elif unit == 'day':
                start = reference_date - timedelta(days=n)
            elif unit == 'week':
                start = reference_date - timedelta(weeks=n)
            else:
                start = reference_date
            
            return start, reference_date
        
        # Specific date patterns
        # Try to parse as a specific date
        try:
            # Handle "on December 6" or "December 6th" etc
            if query_lower.startswith('on '):
                query_lower = query_lower[3:]
            
            parsed_date = date_parser.parse(query_lower, fuzzy=True)
            
            # If no time specified, use whole day
            if parsed_date.hour == 0 and parsed_date.minute == 0:
                start = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1) - timedelta(microseconds=1)
            else:
                # Specific time given, use 1 hour window
                start = parsed_date
                end = parsed_date + timedelta(hours=1)
            
            return start, end
            
        except (ValueError, TypeError):
            pass
        
        # Date range pattern (from X to Y)
        range_pattern = r'from (.+) to (.+)'
        match = re.match(range_pattern, query_lower)
        if match:
            try:
                start_str = match.group(1).strip()
                end_str = match.group(2).strip()
                
                # Recursively parse start and end
                start, _ = DateParser.parse_date_query(start_str, reference_date)
                _, end = DateParser.parse_date_query(end_str, reference_date)
                
                return start, end
            except:
                pass
        
        # Between pattern
        between_pattern = r'between (.+) and (.+)'
        match = re.match(between_pattern, query_lower)
        if match:
            try:
                start_str = match.group(1).strip()
                end_str = match.group(2).strip()
                
                start, _ = DateParser.parse_date_query(start_str, reference_date)
                _, end = DateParser.parse_date_query(end_str, reference_date)
                
                return start, end
            except:
                pass
        
        # Recent videos (last week)
        if query_lower in ['recent', 'recently', 'latest']:
            start = reference_date - timedelta(days=7)
            return start, reference_date
        
        # Default: if we can't parse, return last 24 hours
        start = reference_date - timedelta(days=1)
        return start, reference_date
    
    @staticmethod
    def format_datetime_range(start: datetime, end: datetime) -> str:
        """Format datetime range for display."""
        if start.date() == end.date():
            # Same day
            return f"{start.strftime('%B %d, %Y')} from {start.strftime('%I:%M %p')} to {end.strftime('%I:%M %p')}"
        else:
            # Different days
            return f"{start.strftime('%B %d, %Y %I:%M %p')} to {end.strftime('%B %d, %Y %I:%M %p')}"
    
    @staticmethod
    def get_date_path_components(dt: datetime) -> Tuple[str, str, str]:
        """Get year/month/day components for directory structure."""
        return (
            dt.strftime('%Y'),
            dt.strftime('%m'),
            dt.strftime('%d')
        )