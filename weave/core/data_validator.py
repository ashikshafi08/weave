from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import jsonschema
from .exceptions import ValidationError

class DataValidator(ABC):
    """
    Abstract base class for data validators.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data validator.

        Args:
            config (Dict[str, Any]): Configuration for the data validator.
        """
        self.config = config
        self.rules = config.get('rules', {})

    @abstractmethod
    async def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate generated data.

        Args:
            data (Dict[str, Any]): Data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        pass

    @abstractmethod
    async def get_validation_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a detailed validation report for the data.

        Args:
            data (Dict[str, Any]): Data to validate.

        Returns:
            Dict[str, Any]: Validation report containing details of any validation errors.
        """
        pass

class SchemaValidator(DataValidator):
    """
    Data validator using JSON Schema for validation.
    """

    async def validate(self, data: Dict[str, Any]) -> bool:
        try:
            jsonschema.validate(instance=data, schema=self.rules)
            return True
        except jsonschema.exceptions.ValidationError:
            return False

    async def get_validation_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            jsonschema.validate(instance=data, schema=self.rules)
            return {"valid": True, "errors": []}
        except jsonschema.exceptions.ValidationError as e:
            return {"valid": False, "errors": [str(e)]}

class CustomValidator(DataValidator):
    """
    Data validator using custom validation functions.
    """

    async def validate(self, data: Dict[str, Any]) -> bool:
        for rule_name, rule_func in self.rules.items():
            if not rule_func(data):
                return False
        return True

    async def get_validation_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        for rule_name, rule_func in self.rules.items():
            if not rule_func(data):
                errors.append(f"Failed validation rule: {rule_name}")
        return {"valid": len(errors) == 0, "errors": errors}

class CompositeValidator(DataValidator):
    """
    Data validator that combines multiple validators.
    """

    def __init__(self, config: Dict[str, Any], validators: List[DataValidator]):
        super().__init__(config)
        self.validators = validators

    async def validate(self, data: Dict[str, Any]) -> bool:
        return all(await validator.validate(data) for validator in self.validators)

    async def get_validation_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        reports = [await validator.get_validation_report(data) for validator in self.validators]
        all_errors = [error for report in reports for error in report["errors"]]
        return {"valid": all(report["valid"] for report in reports), "errors": all_errors}

class DataValidatorFactory:
    @staticmethod
    def create_validator(validator_type: str, config: Dict[str, Any]) -> DataValidator:
        """
        Create a DataValidator instance based on the provided type and configuration.

        Args:
            validator_type (str): Type of validator to create.
            config (Dict[str, Any]): Configuration for the validator.

        Returns:
            DataValidator: An instance of a DataValidator subclass.

        Raises:
            ValueError: If an unsupported validator type is specified.
        """
        if validator_type == 'schema':
            return SchemaValidator(config)
        elif validator_type == 'custom':
            return CustomValidator(config)
        elif validator_type == 'composite':
            sub_validators = [DataValidatorFactory.create_validator(v['type'], v['config']) for v in config.get('validators', [])]
            return CompositeValidator(config, sub_validators)
        else:
            raise ValueError(f"Unsupported validator type: {validator_type}")