import requests
import threading
import time

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"

# Helper function to set headers with JWT token
def get_headers(token):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

class TestUser:
    def __init__(self, username, password, role="user"):
        self.username = username
        self.password = password
        self.role = role
        self.token = None
        self.chat_id = None
        self.workspace_id = None

    def test_register(self):
        print(f"[{self.username}] Testing /register endpoint...")
        url = f"{BASE_URL}/register"
        # /register expects form fields, so use data=
        payload = {
            "username": self.username,
            "password": self.password
        }
        response = requests.post(url, data=payload)
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        print(f"[{self.username}] Response Body: {response.json()}\n")
        return response.json()

    def test_login(self):
        print(f"[{self.username}] Testing /login endpoint...")
        url = f"{BASE_URL}/login"
        # /login expects form fields, so use data=
        payload = {
            "username": self.username,
            "password": self.password
        }
        response = requests.post(url, data=payload)
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        response_json = response.json()
        print(f"[{self.username}] Response Body: {response_json}\n")
        if response.status_code == 200 and "access_token" in response_json:
            self.token = response_json["access_token"]  # Save the JWT token
        return response_json

    def update_role(self):
        # This endpoint now expects JSON (username, new_role)
        if self.role == "user":
            return
        print(f"[{self.username}] Updating role to {self.role}...")
        url = f"{BASE_URL}/update-role"
        payload = {
            "username": self.username,
            "new_role": self.role
        }
        # Use json=payload since /update-role expects JSON
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Role update Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Role update Response Body: {response.json()}\n")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            return None

    def test_create_workspace(self):
        # /workspaces now expects JSON (name)
        if self.role not in ["admin", "superadmin"]:
            print(f"[{self.username}] Insufficient permissions to create a workspace.\n")
            return
        print(f"[{self.username}] Testing /workspaces endpoint...")
        url = f"{BASE_URL}/workspaces"
        payload = {"name": f"{self.username}_workspace"}
        # Use json=payload since /workspaces expects JSON
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
            if response.status_code == 200:
                self.workspace_id = response.json().get("workspace_id")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            return None

    def test_assign_user_to_workspace(self, user_id):
        # /workspaces/{workspace_id}/assign-user now expects JSON (user_id)
        if self.role not in ["admin", "superadmin"]:
            print(f"[{self.username}] Insufficient permissions to assign a user to a workspace.\n")
            return
        if self.workspace_id is None:
            print(f"[{self.username}] No workspace_id available.\n")
            return
        print(f"[{self.username}] Testing /workspaces/{self.workspace_id}/assign-user endpoint...")
        url = f"{BASE_URL}/workspaces/{self.workspace_id}/assign-user"
        payload = {"user_id": user_id}
        # Use json=payload since assign-user expects JSON
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            return None

    def test_upload_document(self, scope):
        print(f"[{self.username}] Testing /documents endpoint with scope {scope}...")
        url = f"{BASE_URL}/documents"
        # /documents expects multipart/form-data: files for the file, data for the scope
        files = {'file': ('test_doc.txt', 'This is a test document for scope testing.')}
        data = {'scope': scope}
        response = requests.post(url, files=files, data=data, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            return None

    def test_chats(self):
        print(f"[{self.username}] Testing /chats endpoint...")
        url = f"{BASE_URL}/chats"
        # /chats is a GET request, token in headers only
        response = requests.get(url, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            return None

    def test_chat(self, new_chat=False, chat_id=None):
        print(f"[{self.username}] Testing /chat endpoint (new_chat={new_chat})...")
        url = f"{BASE_URL}/chat"
        # /chat expects JSON for the body
        payload = {
            "query": "How do I connect to the network?",
            "new_chat": new_chat
        }
        if chat_id:
            payload["chat_id"] = chat_id
        response = requests.post(url, json=payload, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print(f"[{self.username}] Response Body: {response_json}\n")
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            response_json = None

        if response_json and "chat_id" in response_json and new_chat:
            self.chat_id = response_json["chat_id"]
        return response_json

    def test_chat_history(self):
        if not self.chat_id:
            print(f"[{self.username}] No chat_id available to fetch history.\n")
            return
        print(f"[{self.username}] Testing /chat/history endpoint...")
        url = f"{BASE_URL}/chat/history/{self.chat_id}"
        response = requests.get(url, headers=get_headers(self.token))
        print(f"[{self.username}] Response Status Code: {response.status_code}")
        try:
            print(f"[{self.username}] Response Body: {response.json()}\n")
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[{self.username}] Non-JSON Response Body: {response.text}\n")
            return None

def user_test_scenario(user):
    user.test_register()
    user.test_login()
    if user.token:
        if user.role in ["admin", "superadmin"]:
            user.update_role()  # Ensure roles are updated after registration
            time.sleep(1)  # Wait briefly for role update to take effect
        if user.role in ["admin", "superadmin"]:
            user.test_create_workspace()
            user.test_assign_user_to_workspace(user_id=2)  # Assign another user
        user.test_upload_document(scope="chat")
        if user.role in ["admin", "superadmin"]:
            user.test_upload_document(scope="workspace")
        if user.role == "superadmin":
            user.test_upload_document(scope="system")
        user.test_chats()
        user.test_chat(new_chat=True)
        user.test_chat(new_chat=False, chat_id=user.chat_id)
        user.test_chat_history()
    else:
        print(f"[{user.username}] Login failed. Unable to test further functionality.")

if __name__ == "__main__":
    users = [
        TestUser("user1", "password1", "user"),
        TestUser("admin1", "password1", "admin"),
        TestUser("superadmin1", "password1", "superadmin")
    ]

    threads = []
    for user in users:
        t = threading.Thread(target=user_test_scenario, args=(user,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
