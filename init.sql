-- Create the 'workspaces' table
CREATE TABLE IF NOT EXISTS workspaces (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE
);

-- Create the 'users' table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',  -- Role of the user (user, admin, superadmin)
    workspace_id INT REFERENCES workspaces(id) -- (Optional) default workspace if needed
);

-- Create a join table for many-to-many relationship between users and workspaces
CREATE TABLE IF NOT EXISTS user_workspaces (
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    workspace_id INT REFERENCES workspaces(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, workspace_id)
);

-- Create the 'user_conversations' table to store chat history
CREATE TABLE IF NOT EXISTS user_conversations (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    chat_id UUID NOT NULL,
    conversation TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create the 'documents' table to store document metadata
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id UUID NOT NULL UNIQUE,
    content TEXT NOT NULL,
    metadata JSONB,
    scope VARCHAR(20) NOT NULL, -- Document scope (chat, profile, workspace, system)
    workspace_id INT REFERENCES workspaces(id), -- Associated workspace
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create the 'api_keys' table for external API key functionality
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    name TEXT,
    key_value TEXT UNIQUE,        -- The actual API key token
    is_global BOOLEAN DEFAULT TRUE,
    workspace_id INT REFERENCES workspaces(id),
    created_by INT REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);
