import datetime
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

from mcp.server.fastmcp import FastMCP

# Create a DateTime MCP Server
mcp = FastMCP("DateTime Operations Tool", port=8001)


@mcp.tool()
def get_local_datetime(format_string: Optional[str] = None) -> str:
    """Get the current local date and time.

    Parameters
    ----------
    format_string : str, optional
        Custom format string for the datetime output. If not provided,
        returns in ISO format. Examples: '%Y-%m-%d %H:%M:%S', '%B %d, %Y at %I:%M %p'

    Returns
    -------
    str
        The current local date and time as a formatted string.

    """
    try:
        now = datetime.datetime.now()

        if format_string:
            return now.strftime(format_string)
        return now.isoformat()
    except Exception as e:
        return f"Error getting local datetime: {str(e)}"


@mcp.tool()
def get_datetime_for_timezone(timezone: str, format_string: Optional[str] = None) -> str:
    """Get the current date and time for any valid timezone.

    Parameters
    ----------
    timezone : str
        Any valid IANA timezone identifier (e.g., 'America/New_York', 'Europe/London',
        'Asia/Tokyo', 'UTC'). Use list_timezones to find available timezones.
    format_string : str, optional
        Custom format string for the datetime output. If not provided,
        returns in ISO format. Examples: '%Y-%m-%d %H:%M:%S', '%B %d, %Y at %I:%M %p'

    Returns
    -------
    str
        The current date and time in the specified timezone as a formatted string.

    """
    try:
        # Validate timezone
        if timezone not in available_timezones():
            return f"Error: '{timezone}' is not a valid timezone. Use list_timezones to see available options."

        # Get current time in the specified timezone
        tz = ZoneInfo(timezone)
        now = datetime.datetime.now(tz)

        if format_string:
            formatted_time = now.strftime(format_string)
        else:
            formatted_time = now.isoformat()

        return f"{formatted_time} ({timezone})"

    except Exception as e:
        return f"Error getting datetime for timezone '{timezone}': {str(e)}"


@mcp.tool()
def list_timezones(search_term: Optional[str] = None) -> str:
    """List available IANA timezone identifiers, optionally filtered by search term.

    Parameters
    ----------
    search_term : str, optional
        Optional search term to filter timezones (case-insensitive).
        For example: 'America', 'Europe', 'New_York', 'London'

    Returns
    -------
    str
        A list of available timezone identifiers, optionally filtered.

    """
    try:
        all_timezones = sorted(available_timezones())

        if search_term:
            search_lower = search_term.lower()
            filtered_timezones = [tz for tz in all_timezones if search_lower in tz.lower()]

            if not filtered_timezones:
                return f"No timezones found matching '{search_term}'"

            result = f"Timezones matching '{search_term}' ({len(filtered_timezones)} found):\n"
            result += "\n".join(f"  • {tz}" for tz in filtered_timezones[:50])  # Limit to 50 results

            if len(filtered_timezones) > 50:
                result += f"\n  ... and {len(filtered_timezones) - 50} more"
        else:
            result = f"All available timezones ({len(all_timezones)} total):\n"
            result += "\n".join(f"  • {tz}" for tz in all_timezones[:100])  # Limit to 100 for readability

            if len(all_timezones) > 100:
                result += f"\n  ... and {len(all_timezones) - 100} more"
                result += "\n\nUse the search_term parameter to filter results."

        return result

    except Exception as e:
        return f"Error listing timezones: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server using HTTP transport
    mcp.run(transport="streamable-http")
