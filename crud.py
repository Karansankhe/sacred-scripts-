import streamlit as st
import sqlite3
import pandas as pd

# SQLite database setup
DB_NAME = "warehouse_management.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create a table for warehouse items
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS warehouse_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price_per_unit REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_item(item_name, quantity, price_per_unit):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO warehouse_items (item_name, quantity, price_per_unit) VALUES (?, ?, ?)",
                   (item_name, quantity, price_per_unit))
    conn.commit()
    conn.close()

def get_items():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM warehouse_items")
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_item(item_id, item_name, quantity, price_per_unit):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE warehouse_items 
        SET item_name = ?, quantity = ?, price_per_unit = ? 
        WHERE id = ?
    """, (item_name, quantity, price_per_unit, item_id))
    conn.commit()
    conn.close()

def delete_item(item_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM warehouse_items WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

# Streamlit App
def main():
    st.title("Warehouse Management System")
    st.sidebar.title("Navigation")
    options = ["Add Item", "View Items", "Update Item", "Delete Item"]
    choice = st.sidebar.selectbox("Choose an action", options)
    
    # Create Database if not exists
    create_database()

    # Add Item
    if choice == "Add Item":
        st.header("Add New Item")
        item_name = st.text_input("Item Name")
        quantity = st.number_input("Quantity", min_value=1, step=1)
        price_per_unit = st.number_input("Price Per Unit", min_value=0.01, step=0.01)
        if st.button("Add Item"):
            if item_name and quantity > 0 and price_per_unit > 0:
                add_item(item_name, quantity, price_per_unit)
                st.success(f"Item '{item_name}' added successfully!")
            else:
                st.error("Please provide valid inputs.")

    # View Items
    elif choice == "View Items":
        st.header("View All Items")
        items = get_items()
        if items:
            df = pd.DataFrame(items, columns=["ID", "Item Name", "Quantity", "Price Per Unit"])
            st.dataframe(df)
        else:
            st.warning("No items found in the warehouse.")

    # Update Item
    elif choice == "Update Item":
        st.header("Update Item Details")
        items = get_items()
        if items:
            df = pd.DataFrame(items, columns=["ID", "Item Name", "Quantity", "Price Per Unit"])
            st.dataframe(df)
            
            item_id = st.number_input("Enter Item ID to Update", min_value=1, step=1)
            item_to_update = [item for item in items if item[0] == item_id]
            if item_to_update:
                item_name = st.text_input("New Item Name", value=item_to_update[0][1])
                quantity = st.number_input("New Quantity", min_value=1, step=1, value=item_to_update[0][2])
                price_per_unit = st.number_input("New Price Per Unit", min_value=0.01, step=0.01, value=item_to_update[0][3])
                
                if st.button("Update Item"):
                    update_item(item_id, item_name, quantity, price_per_unit)
                    st.success(f"Item ID {item_id} updated successfully!")
            else:
                st.warning("Invalid Item ID.")
        else:
            st.warning("No items available to update.")

    # Delete Item
    elif choice == "Delete Item":
        st.header("Delete Item")
        items = get_items()
        if items:
            df = pd.DataFrame(items, columns=["ID", "Item Name", "Quantity", "Price Per Unit"])
            st.dataframe(df)
            
            item_id = st.number_input("Enter Item ID to Delete", min_value=1, step=1)
            if st.button("Delete Item"):
                item_to_delete = [item for item in items if item[0] == item_id]
                if item_to_delete:
                    delete_item(item_id)
                    st.success(f"Item ID {item_id} deleted successfully!")
                else:
                    st.warning("Invalid Item ID.")
        else:
            st.warning("No items available to delete.")

if __name__ == "__main__":
    main()
