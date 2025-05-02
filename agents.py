from collections import OrderedDict

class TeacherAgent:
    
    # constants for the possible teacher actions. For now its just 2 actions
    # but we can have more complex actions later on
    ACTION_SIMPLE = 0
    ACTION_COMPLEX = 1
    
    NUM_ACTIONS = 2     # TODO: make this dynamic based on the length of the action list or whatever
    
    def __init__(self, agent_id):
        """Initializes the teacher agent

        Args:
            agent_id (str): A unique identifier for this agent.
        """
        self.agent_id = agent_id
        
    def reset(self):
        """Resets the teacher's internal state (if any) at the start 
        of an episode
        """
        # No internal state for now so skip
        pass
    
class StudentAgent:
    """Represents the student agent in the environment"""
    # constants for possible student actions - again very simple for now
    ACTION_STUDY = 0
    ACTION_ASK = 1
    
    NUM_ACTIONS = 2     # TODO: make this dynamic based on the length of the action list or whatever
    
    # constants for knowledge levels
    KNOWLEDGE_LOW = 0
    KNOWLEDGE_MEDIUM = 1
    KNOWLEDGE_HIGH = 2
    
    _NUM_KNOWLEDGE_LEVELS = 3 # used for observation space def
    
    def __init__(self, agent_id, initial_knowledge=KNOWLEDGE_LOW):
        """Initializes the student agent

        Args:
            agent_id (str): a unique identifier for this agent
            initial_knowledge (int, optional): The starting knowledge level for this student. Defaults to KNOWLEDGE_LOW.
        """
        
        self.agent_id = agent_id
        self.inital_knowledge = initial_knowledge
        self.knowledge = initial_knowledge
    
    def reset(self):
        """Resets the student's knowledge to the initial level
        """
        self.knowledge = self.inital_knowledge
        # TODO: idk if we really need to do that but lets see
        
    def update_state(self, teacher_action, student_action) -> bool:
        """Updates the student's knowledge state based on teacher and self actions.

        Args:
            teacher_action (int): the action taken by the teacher
            student_action (int): the action taken by this student
            
        Returns:
            bool: True if knowledge increased, False otherwise
        """
        # ! something basic for now. Will likely change this later to be more complex and right
        knowledge_increased = False
        if student_action == self.ACTION_STUDY:
            if teacher_action == TeacherAgent.ACTION_SIMPLE:
                # Simple teaching helps if knowledge is not maxed out
                if self.knowledge < self.KNOWLEDGE_HIGH:
                    self.knowledge += 1
                    knowledge_increased = True
            elif teacher_action == TeacherAgent.ACTION_COMPLEX:
                # Complex teaching helps only if student has medium/high knowledge
                if self.knowledge >= self.KNOWLEDGE_MEDIUM and self.knowledge < self.KNOWLEDGE_HIGH:
                    self.knowledge += 1
                    knowledge_increased = True
        # Asking questions currently doesn't change knowledge state
        elif student_action == self.ACTION_ASK:
            pass

        return knowledge_increased
    
    def get_observation(self): 
        """Returns the student's observable state (their knowledge level)"""
        return self.knowledge

    