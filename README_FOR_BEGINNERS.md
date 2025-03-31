# Solar Panel Doctor: A Super Simple Guide

## Hi there! 👋

This project is like a **doctor for solar panels**. Just like a doctor checks if you're healthy, our system checks if solar panels are working properly!

## What are solar panels?

Solar panels are those big blue or black rectangles you might see on rooftops. They turn sunlight into electricity that can power things in your home like lights, TVs, and computers.

## What does our project do?

Sometimes solar panels get sick! They might:
- Have a broken wire
- Get partially covered by a shadow
- Wear out over time

Our Solar Panel Doctor can tell when a panel is sick and what kind of sickness it has!

## How does it work? (Super simple version)

1. We measure two things from the solar panel:
   - **Current**: How much electricity is flowing (like water in a pipe)
   - **Voltage**: How strong the electricity is (like water pressure)

2. We put these numbers into our smart computer program

3. The program tells us if the panel is:
   - **Healthy**: Working great! 😊
   - **Sick Type 1**: Has a short circuit (electricity taking a shortcut)
   - **Sick Type 2**: Has a broken connection (electricity can't flow)
   - **Sick Type 3**: Partly in shadow (not getting enough sun)
   - **Sick Type 4**: Getting old and tired (not working as well as it used to)

## What's a MySQL database?

Think of a MySQL database as a super organized digital filing cabinet where we store all the information about our solar panels. It helps us keep track of:
- When we checked the panels
- What the measurements were
- If the panels were healthy or sick

MySQL is a powerful system that can handle lots of information at once and is used by many big websites and companies.

## How to use our Solar Panel Doctor

### Step 1: Get the stuff you need
1. **Python**: This is the language our program speaks
2. **MySQL**: This is where we store all our information

### Step 2: Set it up
1. Download this project
2. Install Python and MySQL
3. Run a special command to get everything ready

### Step 3: Start the doctor
1. Open your command prompt (the black window) and type:
   ```
   # For basic functionality
   python app.py --host 127.0.0.1 --port 8080

   # For advanced features with MATLAB integration (if available)
   python solar_fault_detection.py --host 127.0.0.1 --port 8080 --matlab
   ```
2. Open your web browser
3. Go to the special address (like visiting the doctor's office):
   ```
   http://127.0.0.1:8080
   ```

### Step 4: Check your panels
1. The program will show you if your panels are healthy or sick
2. If they're sick, it will tell you what kind of sickness they have
3. It will also suggest what you should do to make them better

## What if something goes wrong?

Don't worry! Check our troubleshooting guide in README_PERSONAL.md. It has simple steps to fix common problems.

## Want to learn more?

If you're curious about how everything works:
- Read README_PERSONAL.md for a more detailed explanation
- Look at README_CODING_GUIDE.md to learn about the programming (if you're interested)

## Remember:

You don't need to understand all the complicated stuff to use our Solar Panel Doctor. Just follow the simple steps, and you'll be taking care of your solar panels like a pro! 🌞
