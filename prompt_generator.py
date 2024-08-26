import sys
import json
import pyperclip

def read_source(fn: str) -> str:
    try:
        with open(fn, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None
    
def copy_to_clipboard(content: str):
    try:
        pyperclip.copy(content)
        print("JSON content copied to clipboard successfully!")
    except Exception as e:
        print(f"Error copying to clipboard: {e}", file=sys.stderr)

def main():
    sources = [
        # "/home/michael/gitrepos/rs_ROMULO/solvers/sbbu/makefile",
        "/home/michael/github/rs_ROMULO/solvers/sbbu/edge.h",
        "/home/michael/github/rs_ROMULO/solvers/sbbu/sbbu.h",
        # ".vscode/launch.json",
        # ".vscode/tasks.json",
    ]

    prompt = """Pls, convert the set_order_indexes_prioritizing_HX_9_HX_distances function to python."""

    # prompt = """
    # The DeviceConfig objects should not have url_data and url_api equal, because it duplicates the data sent to our server. Pls, to fix this, write a script that checks if they are the same and change the url_data to 'api_dummy' as defined in the urls.py. Also, (a) change the DeviceConfig.url_data default value and (b) add a validation process to avoid it to happen again.
    # """

    # prompt = """
    # Pls, convert the post_portdata.py script to a django management command similar to the devices_cmd_api_post_status.py.
    # Add detailed comments and follow design patterns to DRY and better readability.
    # """
    
    # prompt = """
    # Pls, improve the encapsulation of the Device.diagnostic method. It should delegate the status and config info to the DeviceStatus and DeviceConfig classes. Pay attention to the spaces that identify the context of each diagnostic. Just give me the relevant code.
    # """
    
    # prompt = """
    # Pls, improve the readbility and add comments to the command in devices_cmd_device_diagnostic.py.
    # """
    
    # prompt = """
    # Pls, change the command in devices_cmd_device_diagnostic.py. Now, the user can give the device_mac or company_nickname. If the device_mac is given, only the device with this mac should be checked. If company_nickname is given, the all devices of this company should be checked.
    # """
    
    # prompt = """
    # Pls, create a DeviceConfigAdmin class with filter by mac. It will make easier to the user to find a specific DeviceConfig.
    # """
    

    source_list = []
    for source in sources:
        content = read_source(source)
        if content is not None:
            source_list.append({'path': source, 'content': content})
        else:
            print(f"Warning: File '{source}' not found and will be skipped.", file=sys.stderr)

    result = {
        'user_request': prompt + " Write only the necessary and documented code to solve this problem.",
        'sources': source_list
    }

    json_result = json.dumps(result, indent=4)
    
    print(json_result)
    
    fn_prompt = 'prompt_content.json'
    print(f'\nSaving prompt to {fn_prompt}')
    with open(fn_prompt, 'w') as f:
        f.write(json_result)

    copy_to_clipboard(json_result)

if __name__ == "__main__":
    main()
